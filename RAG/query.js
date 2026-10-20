import { OpenAIEmbeddings } from '@langchain/openai';
import { QdrantVectorStore } from '@langchain/qdrant';
import OpenAI from 'openai';
import { env } from '../env.js';

const client = new OpenAI({
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: env.OPENROUTER_API_KEY,
});

async function query(userQuery) {
    /**
     * Convert user query to vector embeddings?
     Initialize the embedding model
    */
    const embeddings = new OpenAIEmbeddings({
        configuration: {
            baseURL: "https://openrouter.ai/api/v1"
        },
        model: 'text-embedding-3-small',
        apiKey: env.OPENROUTER_API_KEY,
    });

    /** search the vectors in the qdrant
     The vector store
    */
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
        embeddings,
        {
            url: 'http://localhost:6333',
            collectionName: 'dsa-docs',
        },
    );

    const vectorRetriver = vectorStore.asRetriever({ k: 5 });
    const results = await vectorRetriver.invoke(userQuery);

    const SYSTEM_PROMPT = `
    You are an expert in answereing user query based on the provided context about document.
    Do not answere anything beyond what is not provided.

    Always also answer the user in short and tell on which page number that content is available and also name of the book

    User Documents:
    ${results.map((e) => JSON.stringify({ bookName: e.metadata.source, pageContent: e.pageContent, pageNumber: e.metadata.loc.pageNumber })).join('\n\n')}
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

await query('who is voldemort');