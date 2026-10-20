import { OpenAIEmbeddings } from '@langchain/openai';
import { QdrantVectorStore } from '@langchain/qdrant';
import OpenAI from 'openai';
import { env } from '../env.ts';

const client = new OpenAI({
    baseURL: "https://openrouter.ai/api/v1",
    apiKey:""
});

async function query(userQuery) {
    const embeddings = new OpenAIEmbeddings({
        configuration: {
            baseURL: "https://openrouter.ai/api/v1"
        },
        model: 'text-embedding-3-small',
        apiKey:""
      }
    )
    console.log("embedding created");
    
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
        embeddings,
        {
            url: 'http://localhost:6333',
            collectionName: 'ai-generated-poem',
        },
    );

    const vectorRetriver = vectorStore.asRetriever({ k: 5 });
    const results = await vectorRetriver.invoke(userQuery);

    const context = results
        .map(
            (e) =>
                `Book/Source: ${e.metadata?.source || 'Unknown'}\nPage: ${e.metadata?.loc?.pageNumber ?? 'N/A'}\nContent:\n${e.pageContent}`
        )
        .join('\n\n---\n\n');

    const SYSTEM_PROMPT = `
    You are an expert in answering user queries based on the provided context about documents.
    Do not answer anything beyond what is provided.

    Always answer the user concisely and tell on which page number (if available) that content is located and the name/source of the book.

    User Documents Context:
    ${context}
  `;

    const llmResponse = await client.chat.completions.create({
        model: 'gpt-4o',
        max_tokens: 500,
        messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: userQuery },
        ],
    });

    console.log(`LLM Response:`, llmResponse.choices[0].message.content);
    console.log(SYSTEM_PROMPT);

}

await query('summarize the poem');