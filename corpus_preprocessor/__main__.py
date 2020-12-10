from .preprocessor import TextProcessor

def main():
    print(f"Running the text processor to pre-process the corpus")
    processor = TextProcessor()
    processor.run()

main()

