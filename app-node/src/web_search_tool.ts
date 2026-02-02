import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";

/**
 * A tool that performs searches using Wikipedia.
 * Very stable and free alternative to standard web search.
 */
export const webSearchTool = tool(
    async ({ query }) => {
        try {
            console.log(`[Tool: Wikipedia Search] Searching for: ${query}`);
            const wikipedia = new WikipediaQueryRun({
                topKResults: 3,
                maxDocContentLength: 4000,
            });
            const result = await wikipedia.invoke(query);
            return result;
        } catch (error: any) {
            console.error(`[Tool: Wikipedia Search] Error: ${error.message}`);
            return `Error performing search: ${error.message}`;
        }
    },
    {
        name: "web_search",
        description: "Search Wikipedia for factual information, history, people, places, and events. Useful for verifying facts outside the PDF context.",
        schema: z.object({
            query: z.string().describe("The search query"),
        }),
    }
);
