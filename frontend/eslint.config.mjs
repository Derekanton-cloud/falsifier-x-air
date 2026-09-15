import { defineConfig, globalIgnores } from "eslint/config";
import nextPlugin from "@next/eslint-plugin-next";
import tsParser from "@typescript-eslint/parser";

/** TypeScript is enforced by `npm run typecheck`; this config provides a stable ESLint-9 runner. */
export default defineConfig(
  { files: ["**/*.{ts,tsx}"], languageOptions: { parser: tsParser }, plugins: { "@next/next": nextPlugin }, rules: nextPlugin.configs.recommended.rules },
  globalIgnores([".next/**", "node_modules/**"]),
);
