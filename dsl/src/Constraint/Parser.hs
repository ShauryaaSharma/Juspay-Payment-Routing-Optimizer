-- | Parser for the routing constraint language.
--
-- Grammar:
--
-- @
--   constraint := "avoid" gateway [ "when" "issuer" issuer ]
--                 "ttl" duration
--                 [ "canary" percent ]
--                 [ "confidence" number ]
--
--   duration   := int ("m" | "h" | "d")
--   percent    := number "%"
-- @
--
-- Examples:
--
-- @
--   avoid PG-Delta when issuer HDFC ttl 8h canary 2% confidence 0.70
--   avoid PG-Bravo ttl 30m
-- @
--
-- Token-level recursive descent over @words@ rather than a character-level
-- parser, and @base@ only rather than megaparsec. The grammar is a handful of
-- keywords; a parser combinator library would be more machinery than the
-- problem has, and keeping the package dependency-free means it builds with a
-- bare GHC.
--
-- Errors carry the offending token and what was expected, because a constraint
-- language that says only "parse error" would be worse than the Python
-- dataclass it replaces.
module Constraint.Parser
  ( ParseError(..)
  , parseConstraint
  , parseConstraints
  , renderParseError
  ) where

import Data.Char (isDigit, isSpace, toUpper)
import Data.List (isSuffixOf)
import Constraint.Types

data ParseError
  = UnexpectedEnd String          -- ^ what was expected
  | Expected String String        -- ^ expected, found
  | BadDuration String
  | BadNumber String String       -- ^ field, token
  | TrailingTokens [String]
  | InvalidValue ValidationError
  deriving (Eq, Show)

renderParseError :: ParseError -> String
renderParseError (UnexpectedEnd expected) =
  "unexpected end of input; expected " ++ expected
renderParseError (Expected expected found) =
  "expected " ++ expected ++ ", found " ++ show found
renderParseError (BadDuration t) =
  "bad duration " ++ show t ++ "; write it as 30m, 8h or 2d"
renderParseError (BadNumber field t) =
  "bad " ++ field ++ " value " ++ show t
renderParseError (TrailingTokens ts) =
  "unexpected trailing input: " ++ unwords ts
renderParseError (InvalidValue e) = renderError e

-- | Parse one constraint from a line.
parseConstraint :: String -> Either ParseError Constraint
parseConstraint input = do
  rest0 <- expect "avoid" (words input)
  (gatewayTok, rest1) <- takeToken "a gateway name" rest0
  (mIssuer, rest2) <- parseIssuer rest1
  rest3 <- expect "ttl" rest2
  (ttlTok, rest4) <- takeToken "a duration such as 8h" rest3
  ttl <- parseDuration ttlTok
  (canaryRaw, rest5) <- parseCanary rest4
  (confidence, rest6) <- parseConfidence rest5
  case rest6 of
    [] -> Right ()
    ts -> Left (TrailingTokens ts)
  canary <- mapLeft InvalidValue (mkCanaryRate canaryRaw)
  let scope = case mIssuer of
        Nothing -> SingleGateway (Gateway gatewayTok)
        Just i  -> IssuerScoped (Gateway gatewayTok) (Issuer (map toUpper i))
  Right Constraint
    { cScope = scope
    , cTtl = ttl
    , cCanary = canary
    , cConfidence = confidence
    }

-- | Parse a whole file. Blank lines and @#@ comments are skipped; the first
-- failure is reported with its line number rather than silently dropped.
parseConstraints :: String -> Either (Int, ParseError) [Constraint]
parseConstraints = go 1 [] . lines
  where
    go _ acc [] = Right (reverse acc)
    go n acc (l:ls)
      | isBlank l || isComment l = go (n + 1) acc ls
      | otherwise = case parseConstraint l of
          Left e  -> Left (n, e)
          Right c -> go (n + 1) (c : acc) ls
    isBlank = all isSpace
    isComment l = case dropWhile isSpace l of
      ('#':_) -> True
      _       -> False

-- Helpers ------------------------------------------------------------------

expect :: String -> [String] -> Either ParseError [String]
expect keyword [] = Left (UnexpectedEnd (show keyword))
expect keyword (t:ts)
  | t == keyword = Right ts
  | otherwise    = Left (Expected (show keyword) t)

takeToken :: String -> [String] -> Either ParseError (String, [String])
takeToken expected []     = Left (UnexpectedEnd expected)
takeToken _        (t:ts) = Right (t, ts)

parseIssuer :: [String] -> Either ParseError (Maybe String, [String])
parseIssuer ("when":rest) = do
  rest1 <- expect "issuer" rest
  (issuer, rest2) <- takeToken "an issuer name" rest1
  Right (Just issuer, rest2)
parseIssuer ts = Right (Nothing, ts)

-- | Canary defaults to 2%, matching the Python constraint layer. A default is
-- correct here: omitting it should give the safe behaviour, not zero, because
-- zero makes the blocked gateway permanently unobservable.
parseCanary :: [String] -> Either ParseError (Double, [String])
parseCanary ("canary":rest) = do
  (tok, rest1) <- takeToken "a percentage such as 2%" rest
  value <- parsePercent tok
  Right (value, rest1)
parseCanary ts = Right (0.02, ts)

parseConfidence :: [String] -> Either ParseError (Double, [String])
parseConfidence ("confidence":rest) = do
  (tok, rest1) <- takeToken "a number between 0 and 1" rest
  case readNumber tok of
    Just v  -> Right (v, rest1)
    Nothing -> Left (BadNumber "confidence" tok)
parseConfidence ts = Right (1.0, ts)

parsePercent :: String -> Either ParseError Double
parsePercent tok
  | "%" `isSuffixOf` tok =
      case readNumber (init tok) of
        Just v  -> Right (v / 100)
        Nothing -> Left (BadNumber "canary" tok)
  | otherwise = Left (BadNumber "canary" tok)

parseDuration :: String -> Either ParseError Duration
parseDuration tok = case reverse tok of
  ('m':ds) -> build Minutes (reverse ds)
  ('h':ds) -> build Hours (reverse ds)
  ('d':ds) -> build Days (reverse ds)
  _        -> Left (BadDuration tok)
  where
    build ctor digits
      | not (null digits) && all isDigit digits = Right (ctor (read digits))
      | otherwise = Left (BadDuration tok)

-- | Accepts @0.7@, @.7@ and @1@. Deliberately narrow: no exponents, no signs.
readNumber :: String -> Maybe Double
readNumber s = case s of
  ('.':_) -> readMaybeDouble ('0' : s)
  _       -> readMaybeDouble s
  where
    readMaybeDouble t
      | null t = Nothing
      | not (all (\c -> isDigit c || c == '.') t) = Nothing
      | length (filter (== '.') t) > 1 = Nothing
      | last t == '.' = Nothing
      | otherwise = Just (read (ensureLeadingDigit t))
    ensureLeadingDigit t = case t of
      ('.':_) -> '0' : t
      _       -> t

mapLeft :: (a -> b) -> Either a c -> Either b c
mapLeft f (Left x)  = Left (f x)
mapLeft _ (Right x) = Right x
