-- | CLI: validate a constraint file, and emit JSON for the Python router.
--
-- @
--   routing-dsl check   constraints.route
--   routing-dsl compile constraints.route > constraints.json
-- @
--
-- Exit codes matter. @check@ returns non-zero on the first invalid constraint
-- so it can sit in CI as a gate, the same way the eval gate does: a constraint
-- file that does not parse should fail the build rather than reach the router.
module Main (main) where

import Data.List (intercalate)
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import System.IO (hPutStrLn, stderr)

import Constraint.Parser
import Constraint.Types

-- | The fleet constraints are checked against. In a deployment this comes from
-- the router's /health endpoint; hardcoded here so the tool has no runtime
-- dependency on a live service and can run in CI.
knownGateways :: [String]
knownGateways = ["PG-Alpha", "PG-Bravo", "PG-Charlie", "PG-Delta", "PG-Echo"]

knownIssuers :: [String]
knownIssuers = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["check", path]   -> run path False
    ["compile", path] -> run path True
    _                 -> usage

usage :: IO ()
usage = do
  hPutStrLn stderr "usage: routing-dsl (check | compile) <file>"
  hPutStrLn stderr ""
  hPutStrLn stderr "  check    validate a constraint file; non-zero exit on error"
  hPutStrLn stderr "  compile  validate, then emit JSON on stdout"
  exitFailure

run :: FilePath -> Bool -> IO ()
run path emitJson = do
  source <- readFile path
  case parseConstraints source of
    Left (line, err) ->
      die (path ++ ":" ++ show line ++ ": " ++ renderParseError err)
    Right parsed -> do
      validated <- mapM checkOne (zip [1 :: Int ..] parsed)
      if emitJson
        then putStrLn ("[" ++ intercalate "," (map toJSON validated) ++ "]")
        else putStrLn (show (length validated) ++ " constraint(s) OK")
      exitSuccess
  where
    checkOne (n, c) = case validate knownGateways knownIssuers c of
      Left err -> die (path ++ ": constraint " ++ show n ++ ": " ++ renderError err)
      Right ok -> return ok

die :: String -> IO a
die msg = do
  hPutStrLn stderr msg
  exitFailure
