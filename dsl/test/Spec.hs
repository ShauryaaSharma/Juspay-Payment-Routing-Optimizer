-- | Property tests for the constraint language.
--
-- Properties rather than examples, because the claims worth making here are
-- universal: /no/ parseable constraint has an out-of-range canary rate, /every/
-- duration converts to a positive number of minutes, /no/ malformed line is
-- silently accepted. QuickCheck generates the counterexample when one exists,
-- which is the whole reason to write the constraint language in Haskell rather
-- than as another Python dataclass.
module Main (main) where

import Control.Monad (unless)
import System.Exit (exitFailure, exitSuccess)
import Test.QuickCheck

import Constraint.Parser
import Constraint.Types

fleet :: [String]
fleet = ["PG-Alpha", "PG-Bravo", "PG-Charlie", "PG-Delta", "PG-Echo"]

issuers :: [String]
issuers = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]

newtype GatewayName = GatewayName String deriving Show
newtype IssuerName = IssuerName String deriving Show

instance Arbitrary GatewayName where
  arbitrary = GatewayName <$> elements fleet

instance Arbitrary IssuerName where
  arbitrary = IssuerName <$> elements issuers

-- | Whatever percentage appears in the source, a constraint that parses always
-- carries a rate inside [0, 1). Out-of-range input must be rejected, never
-- clamped: silently turning @canary 150%@ into 1.0 would produce a constraint
-- that blocks nothing while claiming to block something.
prop_canaryAlwaysInRange :: GatewayName -> Int -> Property
prop_canaryAlwaysInRange (GatewayName g) pct =
  let src = "avoid " ++ g ++ " ttl 1h canary " ++ show pct ++ "%"
  in case parseConstraint src of
       Left _  -> property True
       Right c ->
         let rate = canaryValue (cCanary c)
         in counterexample (src ++ " -> " ++ show rate)
              (rate >= 0 && rate < 1)

-- | Every accepted duration is positive, in whichever unit it was written.
prop_durationsConvertToMinutes :: GatewayName -> Positive Int -> Property
prop_durationsConvertToMinutes (GatewayName g) (Positive n) =
  conjoin
    [ check ("ttl " ++ show n ++ "m") n
    , check ("ttl " ++ show n ++ "h") (n * 60)
    , check ("ttl " ++ show n ++ "d") (n * 60 * 24)
    ]
  where
    check clause expected =
      let src = "avoid " ++ g ++ " " ++ clause
      in counterexample src $ case parseConstraint src of
           Right c -> durationMinutes (cTtl c) == expected
           Left _  -> False

-- | The invariant the type system is carrying: an issuer-scoped constraint
-- always has an issuer, a gateway-wide one never does. There is no third case
-- to test because 'Scope' cannot express one.
prop_scopeCarriesItsPayload :: GatewayName -> IssuerName -> Property
prop_scopeCarriesItsPayload (GatewayName g) (IssuerName i) =
  let scoped = parseConstraint ("avoid " ++ g ++ " when issuer " ++ i ++ " ttl 1h")
      wide   = parseConstraint ("avoid " ++ g ++ " ttl 1h")
  in counterexample (show (scoped, wide)) $ case (scoped, wide) of
       (Right a, Right b) -> isScoped (cScope a) && not (isScoped (cScope b))
       _                  -> False
  where
    isScoped (IssuerScoped _ _) = True
    isScoped _                  = False

-- | Validation is the last barrier before a name reaches the router. The
-- Python side enforces the same rule in @from_diagnosis@; this one catches it
-- earlier, in CI, before the file is ever loaded.
prop_unknownNamesRejected :: Property
prop_unknownNamesRejected = conjoin
  [ rejected "avoid PG-Imaginary ttl 1h"
  , rejected "avoid PG-Alpha when issuer NOT-A-BANK ttl 1h"
  ]
  where
    rejected src = counterexample src $ case parseConstraint src of
      Left _  -> True                       -- rejected at parse time
      Right c -> isLeft (validate fleet issuers c)
    isLeft (Left _) = True
    isLeft _        = False

-- | A well-formed constraint survives validation against the real fleet.
prop_wellFormedAccepted :: GatewayName -> IssuerName -> Property
prop_wellFormedAccepted (GatewayName g) (IssuerName i) =
  let src = "avoid " ++ g ++ " when issuer " ++ i ++ " ttl 8h canary 2% confidence 0.7"
  in counterexample src $ case parseConstraint src of
       Left _  -> False
       Right c -> case validate fleet issuers c of
         Right _ -> True
         Left _  -> False

-- | Malformed input is rejected, not defaulted. Each of these is a mistake
-- someone will actually make.
prop_garbageRejected :: Property
prop_garbageRejected = conjoin (map rejects cases)
  where
    cases =
      [ "avoid"                              -- no gateway
      , "avoid PG-Alpha"                     -- no ttl
      , "avoid PG-Alpha ttl"                 -- ttl with no value
      , "avoid PG-Alpha ttl 8x"              -- unknown unit
      , "avoid PG-Alpha ttl 8h canary 2"     -- percentage without %
      , "avoid PG-Alpha ttl 8h wat"          -- trailing junk
      , "block PG-Alpha ttl 8h"              -- wrong keyword
      , "avoid PG-Alpha when HDFC ttl 8h"    -- missing 'issuer'
      , "avoid PG-Alpha ttl 0h"              -- expires immediately
      ]
    rejects src = counterexample src $ case parseConstraint src of
      Left _  -> True
      Right c -> isLeft (validate fleet issuers c)
    isLeft (Left _) = True
    isLeft _        = False

-- | Comments and blank lines are skipped; a bad line reports its number.
prop_fileParsing :: Property
prop_fileParsing = conjoin
  [ counterexample "valid file" $
      case parseConstraints "# note\n\navoid PG-Alpha ttl 1h\n" of
        Right cs -> length cs == 1
        Left _   -> False
  , counterexample "line number reported" $
      case parseConstraints "avoid PG-Alpha ttl 1h\nnonsense\n" of
        Left (n, _) -> n == 2
        Right _     -> False
  ]

main :: IO ()
main = do
  results <- mapM (quickCheckResult . withMaxSuccess 200)
    [ property prop_canaryAlwaysInRange
    , property prop_durationsConvertToMinutes
    , property prop_scopeCarriesItsPayload
    , property prop_unknownNamesRejected
    , property prop_wellFormedAccepted
    , property prop_garbageRejected
    , property prop_fileParsing
    ]
  unless (all isSuccess results) exitFailure
  exitSuccess
