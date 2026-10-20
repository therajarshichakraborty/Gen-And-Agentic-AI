from llama_cloud import LlamaCloud

client = LlamaCloud()

file = client.files.create(file="document.pdf", purpose="parse")
result = client.parsing.parse(
    file_id=file.id,
    tier="agentic",
    version="latest",
    expand=["markdown"],
)

response = result.markdown.pages[0].markdown

print(response)