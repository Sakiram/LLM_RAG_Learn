import * as dotenv from "dotenv";
import path from "path";

dotenv.config();

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
export const MODEL = process.env.MODEL || "gpt-4o-mini";
export const EMBED_MODEL = "text-embedding-3-small";

// Go up two levels from src (app-node/src -> app-node -> root)
// Actually, if we run from app-node, __dirname will be somewhere in dist or src.
// Let's assume we run from app-node root.
// If we are in src/config.ts, we need to go up to app-node, then up to root.

// process.cwd() is usually where we run the script from.
// If we run `npx ts-node src/index.ts` from `app-node` dir:
export const BASE_DIR = path.resolve(process.cwd(), "..");
export const DATA_DIR = path.join(BASE_DIR, "data");
export const VECTOR_DIR = path.join(BASE_DIR, "vectorstore");

console.log(`DATA_DIR: ${DATA_DIR}`);
console.log(`VECTOR_DIR: ${VECTOR_DIR}`);
