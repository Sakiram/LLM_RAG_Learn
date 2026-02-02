import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export async function loadAndSplitPdf(pdfPath: string) {
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    return await splitter.splitDocuments(docs);
}
