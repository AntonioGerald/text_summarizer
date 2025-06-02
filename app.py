from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__, template_folder='.')

tokenizer = AutoTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("cahya/bert2bert-indonesian-summarization")

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
if request.method == 'POST':
        input_text = request.form['text']
        if len(input_text) > 50:
            inputs = tokenizer.encode("ringkasan: " + input_text, return_tensors="pt", max_length=512, truncation=True)

            summary_ids = model.generate(
                inputs,
                max_length=250,
                min_length=40,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                num_beams=1,            
                num_return_sequences=1
            )


            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            summary = "Masukkan teks lebih dari 50 karakter."
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
