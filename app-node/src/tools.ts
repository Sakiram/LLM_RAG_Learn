import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { VectorStore } from "@langchain/core/vectorstores";

export function buildRetrieverTool(vectorStore: VectorStore, pdfName: string) {
    // k=3 similar to python
    const retriever = vectorStore.asRetriever({ k: 3 });

    // Sanitize name for tool
    const safeName = "search_" + pdfName.replace(".pdf", "").replace(/\s+/g, "_").replace(/-/g, "_").toLowerCase();

    return tool(
        async ({ query }) => {
            const docs = await retriever.invoke(query);
            // docs is an array of Documents
            if (!docs || docs.length === 0) return "No relevant information found.";

            return docs.map((doc, i) => `Source ${i + 1}:\n${doc.pageContent}`).join("\n\n");
        },
        {
            name: safeName,
            description: `Search and retrieve information from ${pdfName}`,
            schema: z.object({
                query: z.string().describe("The query to search the PDF for")
            })
        }
    );
}
