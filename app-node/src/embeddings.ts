import { OpenAIEmbeddings } from "@langchain/openai";
import { OPENAI_API_KEY, EMBED_MODEL } from "./config";

export function getEmbeddings() {
    return new OpenAIEmbeddings({
        modelName: EMBED_MODEL,
        openAIApiKey: OPENAI_API_KEY,
    });
}
