import os
import sys
import torch
import random
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.configs.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
import dash_core_components as dcc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output, State


# GPT 2 model file name downloaded for the project
model_file = 'GPT2/models/gpt2-pytorch_model.bin'

# Parameters for the GPT-2 Model
gpt2_parameters = {
                    "quiet": False,  # Whether or not to print the parameters that are used for input
                    "nsamples": 1,
                    "unconditional": False,  # Output text that does not understand the context of what is input
                    "batch_size": -1,
                    "length": 150,  # The amount of words to predict after input text (max: 512 default: -1)
                    "temperature": 0.9,
                    "top_k": 40,
                    }

# Dash implementation with tab design
app = dash.Dash(__name__)
app.title = 'GPT 2 - Text Generator'
app.layout = html.Div(style={'text-align': 'center', 'width': '45%', 'padding-left': "27.5%"},
                      className='layout-container',
                      children=[html.H1('Transformer Neural Network - GPT 2'), html.H1('Text Generator'),

                                dcc.Tabs(
                                    id="tabs-with-classes",
                                    value='tab-1',
                                    style={'width': '101%', 'font-weight': 'bold'},
                                    parent_className='custom-tabs',
                                    className='custom-tabs-container',
                                    children=[
                                        dcc.Tab(
                                            label='Fake News',
                                            value='tab-1',
                                            className='custom-tab',
                                            selected_className='custom-tab--selected'
                                        ),
                                        dcc.Tab(
                                            label='Q & A',
                                            value='tab-2',
                                            className='custom-tab',
                                            selected_className='custom-tab--selected'
                                        ),
                                        dcc.Tab(
                                            label='Summarizer',
                                            value='tab-3', className='custom-tab',
                                            selected_className='custom-tab--selected'
                                        ),
                                        dcc.Tab(
                                            label='Python Gen',
                                            value='tab-4',
                                            className='custom-tab',
                                            selected_className='custom-tab--selected'
                                        ),
                                    ]),


    dcc.Textarea(id="text-inputer",
                 placeholder="Input Text for Beginning",
                 style={'background-image': 'url(assets/input_data_image.jpg)', 'background-size': 'auto 100%',
                   'background-repeat': 'no-repeat', 'background-position': 'center','font-weight':'bold',
                   'width':"100%", 'height':'250px', 'vertical-align':'top', 'text-align':'left',
                   'border-top': '0px', 'padding-top': '5%'},
                 value='In a shocking finding, scientist discovered a herd of unicorns living in a remote,'
                       ' previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers'
                       ' was the fact that the unicorns spoke perfect English.'),


    html.Br(), html.Br(),
    html.Button(id="submit-text-input",
                children=["Complete Text"],
                style={'backgroundColor': '#FFFFFF',"text-align": "center"}),html.Br(),

    html.Div(children=[html.Br(), html.Div(id="out-all-types")], style={'width': '100%', 'white-space': 'pre-wrap',
                                                                        'text-align': 'left'}), html.Br(), html.Br()])


# GPT 2 Text Generator
if os.path.exists(model_file):
    state_dict = torch.load(model_file, map_location='cpu' if not torch.cuda.is_available() else None)
else:
    print('The file in "model_file" variable does not exist')
    print('Please download one of the following GPT 2 Model suggestions to proceed:')
    print('SMALLEST SIZE: https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin')
    print('SMALL SIZE: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin')
    print('MEDIUM SIZE: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin')
    print('LARGE SIZE: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin')
    print()
    print("Include one of the models above in the project's directories, "
          "or point 'model_file' to the file to proceed")
    sys.exit()


def text_generator(input_text):
    if gpt2_parameters.get("quiet") is False:
        print('GPT-2 parameters used: ' + str(gpt2_parameters))

    if gpt2_parameters.get("batch_size") == -1:
        gpt2_parameters["batch_size"] = 1
    assert gpt2_parameters.get("nsamples") % gpt2_parameters.get("batch_size") == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = get_encoder()



    print(GPT2Config(model_file).output_config())
    config = GPT2Config(model_file)

    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if gpt2_parameters.get("length") == -1:
        gpt2_parameters["length"] = config.n_ctx // 2
    elif gpt2_parameters.get("length") > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print('TEXT INPUT: ' + input_text)
    context_tokens = enc.encode(input_text)

    generated = 0
    for _ in range(gpt2_parameters.get("nsamples") // gpt2_parameters.get("batch_size")):
        out = sample_sequence(
            model=model, length=gpt2_parameters.get("length"),
            context=context_tokens if not gpt2_parameters.get("unconditional") else None,
            start_token=enc.encoder['<|endoftext|>'] if gpt2_parameters.get("unconditional") else None,
            batch_size=gpt2_parameters.get("batch_size"),
            temperature=gpt2_parameters.get("temperature"), top_k=gpt2_parameters.get("top_k"), device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(gpt2_parameters.get("batch_size")):
            generated += 1
            text = enc.decode(out[i])
            context_tokens = enc.encode(text)
            if gpt2_parameters.get("quiet") is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            if '<|endoftext|>' in text:
                print(input_text + text.replace("<|endoftext|>",' (END-OF-TEXT)'))
                return input_text + text.replace("<|endoftext|>",' (END-OF-TEXT)')
            else:
                print(input_text + text + '...')
                return input_text + text + '...'


# Dash Callbacks
@app.callback(Output("out-all-types", "children"),
              [Input("submit-text-input",'n_clicks')],
              [State("text-inputer","value")])
def generate_new_text(n_clicks, entered_text):
    if not n_clicks:
        return
    if (n_clicks % 1) == 0:
        print('Submit Button was Pushed - beginning to predict the rest of the text')

        return text_generator(entered_text)
    else:
        return


@app.callback(Output('text-inputer', 'value'),
              [Input('tabs-with-classes', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return 'In a shocking finding, scientist discovered a herd of unicorns living in a ' \
               'remote, previously unexplored valley, in the Andes Mountains. Even more surprising' \
               ' to the researchers was the fact that the unicorns spoke perfect English.'

    elif tab == 'tab-2':
        return 'Question: What are some applications of AI?\nAnswer: Natural language processing, Chatbots,' \
               ' Sentiment analysis, Sales prediction, Self-driving cars, Facial expression recognition, Image' \
               ' tagging\n\nQuestion: What are some programming languages used in AI?\nAnswer: Python, ' \
               'R, Lisp, Prolog, Java\n\nQuestion: What is the Turing test?\nAnswer: The Turing test is a method to test a ' \
               'machine’s ability to match the human-level intelligence. A machine is used to challenge human ' \
               'intelligence, and when it passes the test it is considered intelligent. Yet a machine could be ' \
               'viewed as intelligent without sufficiently knowing how to mimic a human.\n\nQuestion: What is a fuzzy' \
               ' logic?\nAnswer: Fuzzy logic is a subset of AI; it is a way of encoding human learning ' \
               'for artificial processing. It is a form of many-valued logic. It is represented as ' \
               'IF-THEN rules.\n\nQuestion: Will AI eventually kill all humans?\nAnswer:'

    elif tab == 'tab-3':
        return 'GPT-2 displays a broad set of capabilities, including the ability to ' \
               'generate conditional synthetic text samples of unprecedented quality, ' \
               'where we prime the model with an input and have it generate a lengthy ' \
               'continuation. In addition, GPT-2 outperforms other language models trained ' \
               'on specific domains (like Wikipedia, news, or books) without needing to use ' \
               'these domain-specific training datasets. On language tasks like question ' \
               'answering, reading comprehension, summarization, and translation, GPT-2 ' \
               'begins to learn these tasks from the raw text, using no task-specific training data' \
               '. While scores on these downstream tasks are far from state-of-the-art, they ' \
               'suggest that the tasks can benefit from unsupervised techniques, given sufficient ' \
               '(unlabeled) data and compute.\n\nTL;DR:'

    elif tab == 'tab-4':
        return "for i in range(number_to_sampled):\n\t# Forward step of the network\n\thidden_sample = " \
               "np.tanh(np.dot(x, self.w_ih) + np.dot(sampling_state, self.w_hh))\n\toutput = " \
               "np.dot(hidden_sample, self.w_ho)\n\tprobs = np.exp(output) / np.sum(np.exp(output)" \
               ")\n\t# Next we find the index with the highest prob\n\tindex = np.random.choice" \
               "(range(self.vocab_size), p=probs.ravel())\n\t# setting x-one_hot_vector for the next" \
               " character\n\tx = np.zeros((1, self.vocab_size))\n\tx[0][index] = 1\n\t# Find " \
               "the char with the sampled index and concat to the output string\n\tchar = self.idx_" \
               "to_vocab[index]\n\tsampled_string += char\n\nmodel_numbers = [5,30]"


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)
