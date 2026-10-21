from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import WebSearch
from pydantic_ai_harness import Advisor, Coder

agent = Agent(
    'anthropic:claude-fable-5-1',
    capabilities=[
        Coder(),
        WebSearch(),
        Advisor('openai:gpt-6.1-sol'),
    ],
)

roulette_agent = Agent(  
    'openai:gpt-6.1-sol',
    deps_type=int,
    output_type=bool,
    instructions=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'

success_number = 18  
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)  

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)

agent.to_cli_sync()