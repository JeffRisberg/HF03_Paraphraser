from flask import Flask, jsonify, request
from transformers import *


model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

app = Flask(__name__)

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")

  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )

  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

@app.route("/paraphrase", methods=["GET"])
def message():
    posted_data = request.get_json()
    phrase = posted_data['phrase']
    paraphrases = get_paraphrased_sentences(model, tokenizer, phrase, num_beams=20, num_return_sequences=20)
    return jsonify(paraphrases)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
