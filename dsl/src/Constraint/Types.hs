-- | The routing constraint language, as types.
--
-- The reason this exists in Haskell rather than as another Python dataclass:
-- most of what @from_diagnosis@ checks at runtime is expressible in a type.
-- A constraint that names an issuer while claiming fleet-wide scope, or
-- carries a canary rate of 1.4, or expires before it starts, should not be
-- representable -- and where it must be checked, the check should happen once,
-- at parse time, not on every routing decision.
--
-- The pipeline is: text -> AST -> validated constraint -> JSON, consumed by
-- the Python router. Haskell owns the grammar and the invariants; Python owns
-- the hot path.
module Constraint.Types
  ( Scope(..)
  , Gateway(..)
  , Issuer(..)
  , Duration(..)
  , CanaryRate
  , Constraint(..)
  , ValidationError(..)
  , mkCanaryRate
  , canaryValue
  , validate
  , durationMinutes
  , renderError
  , toJSON
  ) where

import Data.List (intercalate)

-- | A gateway name. Newtyped so it cannot be confused with an issuer name,
-- which is a mistake that is easy to make and silent when both are @String@.
newtype Gateway = Gateway { unGateway :: String } deriving (Eq, Show)

newtype Issuer = Issuer { unIssuer :: String } deriving (Eq, Show)

-- | Scope carries its own payload, so an illegal combination cannot be built.
-- @FleetWide@ has nowhere to put an issuer; @IssuerScoped@ requires one.
-- This is the invariant `from_diagnosis` enforces with a runtime check.
data Scope
  = SingleGateway Gateway
  | IssuerScoped Gateway Issuer
  deriving (Eq, Show)

-- | Durations are minutes, but written as @30m@ / @8h@ / @2d@ so the source
-- text says what it means.
data Duration = Minutes Int | Hours Int | Days Int deriving (Eq, Show)

durationMinutes :: Duration -> Int
durationMinutes (Minutes n) = n
durationMinutes (Hours n)   = n * 60
durationMinutes (Days n)    = n * 60 * 24

-- | A canary rate in [0, 1). Abstract on purpose: the only constructor is
-- 'mkCanaryRate', so a value of this type is always in range.
newtype CanaryRate = CanaryRate Double deriving (Eq, Show)

-- | Read the rate back out. Exported so tests can state the range law as a
-- property; the type still has only one constructor, so the law holds by
-- construction and the property is a check on the parser, not on arithmetic.
canaryValue :: CanaryRate -> Double
canaryValue (CanaryRate r) = r

unCanary :: CanaryRate -> Double
unCanary = canaryValue

-- | Rejects out-of-range rates, and rejects 1.0 specifically: a constraint
-- that lets everything through is not a constraint.
mkCanaryRate :: Double -> Either ValidationError CanaryRate
mkCanaryRate r
  | r < 0     = Left (CanaryOutOfRange r)
  | r >= 1    = Left (CanaryOutOfRange r)
  | otherwise = Right (CanaryRate r)

data Constraint = Constraint
  { cScope      :: Scope
  , cTtl        :: Duration
  , cCanary     :: CanaryRate
  , cConfidence :: Double
  } deriving (Eq, Show)

data ValidationError
  = CanaryOutOfRange Double
  | NonPositiveTtl Int
  | ConfidenceOutOfRange Double
  | ZeroCanaryWithoutOverride
  | UnknownGateway String [String]
  | UnknownIssuer String [String]
  deriving (Eq, Show)

renderError :: ValidationError -> String
renderError (CanaryOutOfRange r) =
  "canary rate " ++ show r ++ " is outside [0, 1); a constraint that lets all \
  \traffic through is not a constraint"
renderError (NonPositiveTtl n) =
  "ttl of " ++ show n ++ " minutes: constraints must expire in the future"
renderError (ConfidenceOutOfRange c) =
  "confidence " ++ show c ++ " is outside [0, 1]"
renderError ZeroCanaryWithoutOverride =
  "canary rate 0 makes the blocked gateway permanently unobservable, so the \
  \constraint could never be retired on evidence; write `canary 0% unsafe` if \
  \that is genuinely intended"
renderError (UnknownGateway g known) =
  "unknown gateway " ++ show g ++ "; known: " ++ intercalate ", " known
renderError (UnknownIssuer i known) =
  "unknown issuer " ++ show i ++ "; known: " ++ intercalate ", " known

-- | Checks that need a value, not just a shape: range bounds, and that the
-- named gateway and issuer actually exist in the fleet.
validate :: [String] -> [String] -> Constraint -> Either ValidationError Constraint
validate gateways issuers c = do
  checkGateway (scopeGateway (cScope c))
  checkIssuer (scopeIssuer (cScope c))
  let ttl = durationMinutes (cTtl c)
  if ttl <= 0 then Left (NonPositiveTtl ttl) else Right ()
  let conf = cConfidence c
  if conf < 0 || conf > 1 then Left (ConfidenceOutOfRange conf) else Right ()
  Right c
  where
    checkGateway (Gateway g)
      | g `elem` gateways = Right ()
      | otherwise         = Left (UnknownGateway g gateways)
    checkIssuer Nothing = Right ()
    checkIssuer (Just (Issuer i))
      | i `elem` issuers = Right ()
      | otherwise        = Left (UnknownIssuer i issuers)

scopeGateway :: Scope -> Gateway
scopeGateway (SingleGateway g)  = g
scopeGateway (IssuerScoped g _) = g

scopeIssuer :: Scope -> Maybe Issuer
scopeIssuer (SingleGateway _)   = Nothing
scopeIssuer (IssuerScoped _ i)  = Just i

-- | Emitted for the Python side. Hand-rolled rather than pulling in aeson:
-- the shape is four scalar fields and a nullable string, and keeping the
-- package dependency-free (base only) means it builds anywhere with GHC.
toJSON :: Constraint -> String
toJSON c = "{" ++ intercalate "," fields ++ "}"
  where
    fields =
      [ kv "gateway" (quoted (unGateway (scopeGateway (cScope c))))
      , kv "issuer"  (maybe "null" (quoted . unIssuer) (scopeIssuer (cScope c)))
      , kv "scope"   (quoted (scopeName (cScope c)))
      , kv "ttl_minutes" (show (durationMinutes (cTtl c)))
      , kv "canary_rate" (show (unCanary (cCanary c)))
      , kv "confidence"  (show (cConfidence c))
      ]
    kv k v = quoted k ++ ":" ++ v
    quoted s = "\"" ++ concatMap esc s ++ "\""
    esc '"'  = "\\\""
    esc '\\' = "\\\\"
    esc ch   = [ch]
    scopeName (SingleGateway _)  = "single_gateway"
    scopeName (IssuerScoped _ _) = "issuer_specific"
