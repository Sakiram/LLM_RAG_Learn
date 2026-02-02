import { StateGraph, END } from "@langchain/langgraph";
import { BaseMessage, SystemMessage, ToolMessage, AIMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { StructuredTool } from "@langchain/core/tools";

// State definition
interface AgentState {
    messages: BaseMessage[];
}

export function buildGraph(llm: ChatOpenAI, tools: StructuredTool[]) {
    // LLM with tools
    const llmWithTools = llm.bindTools(tools);
    const toolsMap = Object.fromEntries(tools.map(t => [t.name, t]));

    // Node definitions
    async function callLlm(state: AgentState) {
        const messages = state.messages;
        const systemMessage = new SystemMessage("Answer questions using the PDF. Cite content.");

        // Combine system message with history for the prompt
        // Note: We don't verify if system message is already in history because we construct it here.
        const response = await llmWithTools.invoke([systemMessage, ...messages]);
        return { messages: [response] };
    }

    async function takeAction(state: AgentState) {
        const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
        if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
            return { messages: [] };
        }

        const results = [];
        for (const call of lastMessage.tool_calls) {
            const tool = toolsMap[call.name];
            if (tool) {
                // tool.invoke expects the args directly usually, but check signature
                // structured tool invoke
                const output = await tool.invoke(call.args);
                results.push(new ToolMessage({
                    tool_call_id: call.id!,
                    name: call.name,
                    content: typeof output === 'string' ? output : JSON.stringify(output)
                }));
            }
        }
        return { messages: results };
    }

    function shouldContinue(state: AgentState) {
        const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
        if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
            return "tools";
        }
        return END;
    }

    const workflow = new StateGraph<AgentState>({
        channels: {
            messages: {
                reducer: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
                default: () => [],
            }
        }
    })
        .addNode("llm", callLlm)
        .addNode("tools", takeAction)
        .addEdge("tools", "llm")
        .addConditionalEdges("llm", shouldContinue)
        .setEntryPoint("llm");

    return workflow.compile();
}
