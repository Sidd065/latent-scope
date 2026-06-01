summarize_system_prompt = """
You are a helpful assistant who's job it is to summarize lists of items with a short label.
You will be presented with a list of items and your job is to choose a label that best summarizes the theme of the list so that someone browsing the labels will have a good idea of what is in the list.
"""


def summarize(items, context=""):
    item_lines = "\n".join(f"<Item>{item}</Item>" for item in items)
    return f"""
You will be presented with a list of items and your job is to summarize the items into a short label of no more than 4 words.
The items come from a natural language processing task that has grouped them by similaritiy.
Since the items in this list are expected to be similar in some ways, the label will be used to distinguish this list from others, so pay attention to what makes a group of similar items distinct.
{context}

Items
--------

{item_lines}

Task
--------
Choose a label that best summarizes the theme of the list so that someone browsing the labels will have a good idea of what is in the list.
Do not use punctuation, Do not explain yourself, respond with only a few words that summarize the list.

Label:"""
