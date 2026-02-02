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
        console.log("\n--- [Agent: Thinking] ---");
        const messages = state.messages;
        const systemMessage = new SystemMessage("Answer questions using tools if needed. If using a PDF tool, cite content. Be concise.");

        const response = await llmWithTools.invoke([systemMessage, ...messages]);

        if (response.content) {
            console.log(`[Agent: Response] ${response.content.toString().substring(0, 100)}${response.content.toString().length > 100 ? '...' : ''}`);
        }

        return { messages: [response] };
    }

    async function takeAction(state: AgentState) {
        const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
        if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
            return { messages: [] };
        }

        const results = [];
        console.log(`--- [Agent: Taking Action] Using ${lastMessage.tool_calls.length} tool(s) ---`);

        for (const call of lastMessage.tool_calls) {
            const tool = toolsMap[call.name];
            if (tool) {
                console.log(`[Tool Call: ${call.name}] Args: ${JSON.stringify(call.args)}`);
                const output = await tool.invoke(call.args);
                const content = typeof output === 'string' ? output : JSON.stringify(output);

                console.log(`[Tool Result: ${call.name}] Output length: ${content.length} chars`);

                results.push(new ToolMessage({
                    tool_call_id: call.id!,
                    name: call.name,
                    content: content
                }));
            } else {
                console.warn(`[Agent: Warning] Tool ${call.name} not found!`);
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
