#!/usr/bin/env python3
# Simple example of using the AI Text-to-SQL Assistant

from ai_text2sql_assistant import AIText2SQLAssistant

def main():
    # Initialize the assistant
    assistant = AIText2SQLAssistant()

    # Define sample schema
    schema = {
        "tables": [
            {"name": "users", "columns": ["id", "name", "email", "created_at"]},
            {"name": "orders", "columns": ["id", "user_id", "amount", "order_date"]},
            {"name": "products", "columns": ["id", "name", "price", "category"]}
        ]
    }

    # Sample queries
    queries = [
        "Show me all users",
        "Get total sales by user",
        "Find orders from last month",
        "What are the top selling products?",
        "Count active users this year"
    ]

    print("AI Text-to-SQL Assistant Example")
    print("=" * 40)

    for query in queries:
        print(f"\nQuery: {query}")

        # Process with different models
        for model in ["gpt4", "sqlformer", "custom_finetuned"]:
            try:
                result = assistant.process_query(query, schema, model)
                print(f"  {model}: {result.sql_query}")
                print(f"    Confidence: {result.confidence:.2f}")
            except Exception as e:
                print(f"  {model}: Error - {e}")

    print("\nExample completed!")

if __name__ == "__main__":
    main()
