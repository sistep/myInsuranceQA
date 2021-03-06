import nltk
# nltk.download()
line="--> Coverage follows the car. Example 1: If you were given a car (loaned) and the car has no insurance, you can buy insurance on the car and your insurance will be primary. Another option, someone helped you to buy a car. For example your credit score isn't good enough to finance, so a friend of yours signed under your loan as a primary payor. You can get insurance under your name and even list your friend on the policy as a loss payee. In this case, we always suggest you get a loan gap coverage: the difference between the car's actual cash value and the amount still owned on it. Example 2: The car you are loaned has insurance. You can buy a policy under your name, list the car on that policy and in case of the accident, your policy will become a secondary or excess. Once the limits of the primary car insurance are exhausted, your coverage would kick in and hopefully pay for the rest. I specifically used the word hopefully, because each accident is unique and it's hard to interpret the coverage without the actual claim scenario. And even with a given claim scenario, sometimes there are 2 possible outcomes of a claim."
tokens=nltk.word_tokenize(line.lower())
print(tokens)
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*','<','>','-']
for punc in english_punctuations:
    pass

# wordEngStop = nltk.corpus.stopwords.words('english')
# print(wordEngStop)