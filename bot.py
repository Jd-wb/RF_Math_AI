from NanoBot import Arithmetic, WebMath, MathEngine, MathMemory, MathScraper, MathLearner

class Bot:
    def __init__(self):
        self.arithmetic = Arithmetic()

    def process_question(self, question: str):
        try:
            result = self.arithmetic.solve_any_math(question)
            return self.format_response(result)
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def format_response(self, result):
        if isinstance(result, dict):
            if 'error' in result:
                return f"Error: {result['error']}"
            if 'results' in result:
                return self.format_multiple_results(result['results'])
            if 'result' in result:
                return f"Result: {result['result']}"
        return f"Result: {result}"

    def format_multiple_results(self, results):
        formatted = ["Solutions found:"]
        for i, res in enumerate(results, 1):
            formatted.append(f"\nSolution {i}:")
            if isinstance(res, dict):
                for key, value in res.items():
                    formatted.append(f"{key}: {value}")
            else:
                formatted.append(str(res))
        return "\n".join(formatted)

if __name__ == "__main__":
    bot = Bot()
    print("Math Problem Solver")
    print("Type 'exit' to quit")
    print("-" * 40)
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not question:
            continue
            
        response = bot.process_question(question)
        print("\n" + response)
