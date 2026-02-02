import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { create, all } from "mathjs";

const math = create(all);

/**
 * A tool that performs mathematical calculations using mathjs.
 */
export const calculatorTool = tool(
    async ({ expression }) => {
        try {
            console.log(`[Tool: Calculator] Evaluating: ${expression}`);
            const result = math.evaluate(expression);
            return typeof result === 'string' ? result : JSON.stringify(result);
        } catch (error: any) {
            console.error(`[Tool: Calculator] Error: ${error.message}`);
            return `Error evaluating expression: ${error.message}`;
        }
    },
    {
        name: "calculator",
        description: "Perform mathematical calculations. Supports arithmetic, functions, and more.",
        schema: z.object({
            expression: z.string().describe("The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"),
        }),
    }
);
