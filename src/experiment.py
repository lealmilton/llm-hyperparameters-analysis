import asyncio
from tqdm import tqdm
import aiohttp
from src.api_utils import get_response

async def run_experiment(prompts, temperatures, top_ps, presence_penalties, frequency_penalties, n, desc):
    results = []
    total_combinations = len(prompts) * len(temperatures) * len(top_ps) * len(presence_penalties) * len(frequency_penalties)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            for temp in temperatures:
                for top_p in top_ps:
                    for presence_penalty in presence_penalties:
                        for frequency_penalty in frequency_penalties:
                            task = asyncio.create_task(get_response(prompt, temp, top_p, presence_penalty, frequency_penalty, n))
                            tasks.append((prompt, temp, top_p, presence_penalty, frequency_penalty, task))
        
        for prompt, temp, top_p, presence_penalty, frequency_penalty, task in tqdm(tasks, total=total_combinations, desc=desc):
            outputs = await task
            results.append({
                "prompt": prompt,
                "temperature": temp,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "outputs": outputs
            })
    return results
