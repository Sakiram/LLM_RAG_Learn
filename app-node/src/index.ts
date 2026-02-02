import readline from "readline";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { MODEL, OPENAI_API_KEY } from "./config";
import { VectorizationManager } from "./vectorization_manager";
import { buildRetrieverTool } from "./tools";
import { buildGraph } from "./graph";
import { VectorStore } from "@langchain/core/vectorstores";

async function main() {
    const manager = new VectorizationManager();
    let currentPdf: string | null = null;
    let currentVectorStore: VectorStore | null = null;
    let graph: any = null;

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: "\n> "
    });

    console.log("\n" + "=".repeat(60));
    console.log("📚 PDF Question & Answer System (Node.js)");
    console.log("=".repeat(60));

    const availablePdfs = await manager.listAvailablePdfs();
    if (availablePdfs.length > 0) {
        console.log("\n📁 Available PDFs in data folder:");
        availablePdfs.forEach(pdf => console.log(`   • ${pdf}`));
    } else {
        console.log("\n⚠️  No PDFs found in data folder");
    }

    console.log("\n💡 Commands:");
    console.log("   • vectorize <filename> - Vectorize a PDF");
    console.log("   • list - Show available PDFs");
    console.log("   • current - Show currently loaded PDF");
    console.log("   • exit/quit - Exit the program");
    console.log("   • Or just ask a question about the loaded PDF");
    console.log("=".repeat(60));

    rl.prompt();

    rl.on("line", async (line) => {
        const userInput = line.trim();
        if (!userInput) {
            rl.prompt();
            return;
        }

        try {
            if (["exit", "quit"].includes(userInput.toLowerCase())) {
                console.log("\n👋 Goodbye!");
                rl.close();
                process.exit(0);
            }

            if (userInput.toLowerCase() === "list") {
                const pdfs = await manager.listAvailablePdfs();
                if (pdfs.length > 0) {
                    console.log("\n📁 Available PDFs:");
                    pdfs.forEach(pdf => console.log(`   • ${pdf}`));
                } else {
                    console.log("\n⚠️  No PDFs found in data folder");
                }
            } else if (userInput.toLowerCase() === "current") {
                if (currentPdf) {
                    console.log(`\n📄 Currently loaded: ${currentPdf}`);
                } else {
                    console.log("\n⚠️  No PDF currently loaded. Use 'vectorize <filename>' first.");
                }
            } else if (userInput.toLowerCase().startsWith("vectorize ")) {
                const pdfName = userInput.substring(10).trim();
                if (!pdfName) {
                    console.log("\n⚠️  Please provide a PDF filename. Example: vectorize myfile.pdf");
                } else {
                    try {
                        const result = await manager.vectorizePdf(pdfName);
                        console.log(`\n${result.message}`);

                        currentPdf = pdfName.endsWith('.pdf') ? pdfName : pdfName + '.pdf';
                        currentVectorStore = result.vectorStore;

                        // Build graph
                        const retrieverTool = buildRetrieverTool(currentVectorStore!, currentPdf);
                        const llm = new ChatOpenAI({
                            modelName: MODEL,
                            openAIApiKey: OPENAI_API_KEY,
                            temperature: 0
                        });

                        graph = buildGraph(llm, [retrieverTool]);
                        console.log(`✓ Ready to answer questions about '${currentPdf}'`);

                    } catch (e: any) {
                        console.error(`\n❌ Error: ${e.message}`);
                    }
                }
            } else {
                // General question
                if (!currentPdf || !graph) {
                    console.log("\n⚠️  Please vectorize a PDF first using: vectorize <filename>");
                } else {
                    console.log(`\n🤔 Searching '${currentPdf}'...`);
                    const result = await graph.invoke({
                        messages: [new HumanMessage(userInput)]
                    });

                    console.log("\n" + "=".repeat(60));
                    console.log("💬 ANSWER:");
                    console.log("=".repeat(60));
                    const lastMsg = result.messages[result.messages.length - 1];
                    console.log(lastMsg.content);
                    console.log("=".repeat(60));
                }
            }

        } catch (error: any) {
            console.error(`\n❌ Error: ${error.message}`);
        }

        rl.prompt();
    });
}

main().catch(console.error);
