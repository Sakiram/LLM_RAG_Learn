import fs from "fs";
import path from "path";
import { DATA_DIR } from "./config";
import { loadAndSplitPdf } from "./pdf_loader";
import { getVectorStore, vectorStoreExists } from "./vector_db";

export class VectorizationManager {
    constructor() {
        // Ensure data dir logic if needed
    }

    private getCollectionName(pdfName: string): string {
        return pdfName.replace(".pdf", "").replace(/\s+/g, "_").replace(/-/g, "_").toLowerCase();
    }

    private getPdfPath(pdfName: string): string {
        if (!pdfName.endsWith(".pdf")) {
            pdfName += ".pdf";
        }
        return path.join(DATA_DIR, pdfName);
    }

    public async listAvailablePdfs(): Promise<string[]> {
        if (!fs.existsSync(DATA_DIR)) return [];
        const files = await fs.promises.readdir(DATA_DIR);
        return files.filter(f => f.toLowerCase().endsWith(".pdf"));
    }

    public async vectorizePdf(pdfName: string, force: boolean = false) {
        if (!pdfName.endsWith(".pdf")) pdfName += ".pdf";

        const collectionName = this.getCollectionName(pdfName);
        const pdfPath = this.getPdfPath(pdfName);

        if (!fs.existsSync(pdfPath)) {
            throw new Error(`PDF file not found: ${pdfPath}`);
        }

        if (!force && vectorStoreExists(collectionName)) {
            const vectorStore = await getVectorStore(undefined, collectionName);
            return { vectorStore, message: `✓ Loaded existing vectorstore for '${pdfName}'` };
        }

        console.log(`📄 Loading and splitting PDF: ${pdfName}`);
        const docs = await loadAndSplitPdf(pdfPath);

        console.log(`🔄 Vectorizing ${docs.length} document chunks...`);
        const vectorStore = await getVectorStore(docs, collectionName);

        return { vectorStore, message: `✓ Successfully vectorized '${pdfName}' (${docs.length} chunks)` };
    }
}
