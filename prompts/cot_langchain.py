import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo")

## Basic Chain of Thought Prompting
standard_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely: {question}.",
)

cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question step by step concisely: {question}",
)

standard_chain = standard_prompt | llm
cot_chain = cot_prompt | llm

question = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"

standard_response = standard_chain.invoke(question).content
cot_response = cot_chain.invoke(question).content

print("Standard Response:")
print(standard_response)
print("\n\nChain of Thought Response:")
print(cot_response)


## Advanced Chain of Thought Techniques
advanced_cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Solve the following problem step by step. For each step:
1. State what you're going to calculate
2. Write the formula you'll use (if applicable)
3. Perform the calculation
4. Explain the result

Question: {question}

Solution:""",
)

advanced_cot_chain = advanced_cot_prompt | llm

complex_question = "A car travels 150 km at 60 km/h, then another 100 km at 50 km/h. What is the average speed for the entire journey?"

advanced_cot_response = advanced_cot_chain.invoke(complex_question).content
print(advanced_cot_response)

## Comparative Analysis

challenging_question = """
A cylindrical water tank with a radius of 1.5 meters and a height of 4 meters is 2/3 full. 
If water is being added at a rate of 10 liters per minute, how long will it take for the tank to overflow? 
Give your answer in hours and minutes, rounded to the nearest minute. 
(Use 3.14159 for π and 1000 liters = 1 cubic meter)"""

standard_response = standard_chain.invoke(challenging_question).content
cot_response = advanced_cot_chain.invoke(challenging_question).content

print("Standard Response:")
print(standard_response)
print("\nChain of Thought Response:")
print(cot_response)

## PRoblem Solving using CoT

logical_reasoning_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""Analyze the following logical puzzle thoroughly. Follow these steps in your analysis:

List the Facts:

Summarize all the given information and statements clearly.
Identify all the characters or elements involved.
Identify Possible Roles or Conditions:

Determine all possible roles, behaviors, or states applicable to the characters or elements (e.g., truth-teller, liar, alternator).
Note the Constraints:

Outline any rules, constraints, or relationships specified in the puzzle.
Generate Possible Scenarios:

Systematically consider all possible combinations of roles or conditions for the characters or elements.
Ensure that all permutations are accounted for.
Test Each Scenario:

For each possible scenario:
Assume the roles or conditions you've assigned.
Analyze each statement based on these assumptions.
Check for consistency or contradictions within the scenario.
Eliminate Inconsistent Scenarios:

Discard any scenarios that lead to contradictions or violate the constraints.
Keep track of the reasoning for eliminating each scenario.
Conclude the Solution:

Identify the scenario(s) that remain consistent after testing.
Summarize the findings.
Provide a Clear Answer:

State definitively the role or condition of each character or element.
Explain why this is the only possible solution based on your analysis.
Scenario:

{scenario}

Analysis:""",
)

logical_reasoning_chain = logical_reasoning_prompt | llm

logical_puzzle = """In a room, there are three people: Amy, Bob, and Charlie. 
One of them always tells the truth, one always lies, and one alternates between truth and lies. 
Amy says, 'Bob is a liar.' 
Bob says, 'Charlie alternates between truth and lies.' 
Charlie says, 'Amy and I are both liars.' 
Determine the nature (truth-teller, liar, or alternator) of each person."""

logical_reasoning_response = logical_reasoning_chain.invoke(logical_puzzle).content
print(logical_reasoning_response)
