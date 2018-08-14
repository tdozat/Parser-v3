# Parser-v3
Stanford CoNLL 2018 Graph-based Dependency Parser

This repo contains the code used for the semantic dependency parser in Dozat & Manning (2018), [Simpler but More Accurate Semantic Dependency Parsing](https://arxiv.org/abs/1807.01396) and for the tagger and parser in Qi, Dozat, Zhang and Manning (2018), [Universal Dependency Parsing from Scratch]().

## Why version 3?
In Parser-v2, I made a few coding choices early on that made some things simple and elegant but made other things really hard to extend. The principle example of this is that the information in each conllu file is more or less stored as a single numpy array. This is fine when each vocabulary type has a fixed number of columns--for example, the word multivocab has three columns, one for storing the pretrained embedding indices, one for the token embedding indices, and one for the subtoken embedding indices (which pointed to a row in another array). Since it was designed for dependency trees, where each token has exactly one head, head information could be stored in one column of the conllu array. In semantic dependency frameworks and enhanced UD, words can have multiple (or no) heads, so the head information can no longer be stored in a single column. Pointing the head to a separate array (like the subtoken index) would mean treating that column separately for the syntactic and semantic representations, which is getting into needlessly hacky territory. The single-array setup is deeply integrated into the code, and I wasn't really happy with the directory structure or how the subtoken models were accessed, so I decided it would be best to start over. This version has a more modular data representation that makes it easy to extend the code to represent more complex annotations like universal features (UFeats) and graph-structured dependencies. I think the directory structure is a bit clearer and a few of the messy files are easier to understand (although one or two are still pretty opaque...). It's not quite production-ready, but if you want to use it for your own research, hopefully this version will be easier to play with!

I'll also try to be more responsive to git issues this time around...

## Dependencies
It runs on Python3 and needs TensorFlow version 1.4 (which uses CUDA 8) or TensorFlow version 1.5--current (which use CUDA 9), although the later TF versions will throw some warnings about deprecated function arguments.

Currently it's in TensorFlow, but my collaborators for the CoNLL18 shared task wrote all their code in PyTorch. When deciding how we were going to consolidate our codebases I was outvoted 2-to-1, so if you want a PyTorch version, one is hopefully on the way!

## How to use
### Training
I principle, a basic dependency parser can be trained by simply running
```bash
python3 main.py train ParserNetwork
```
This will load the parameter configuration in `config/defaults.cfg`.
If you want to train a tagger instead, use `train TaggerNetwork`, and if you want to train a graph-structured dependencies scheme (e.g. semantic dependencies), use `train GraphParserNetwork`.
```bash
python3 main.py train {TaggerNetwork/ParserNetwork/GraphParserNetwork}
```
**NB** Currently you need to pipeline things--if you want your parser to read from tags predicted by a tagger you trained, you need to tag the files and then feed those into the parser. I've started some code for tagging and parsing without having to spit intermediate predictions out to disk, but it's not done yet.

**NB** This is at its core a Universal Dependencies parsing system, and it expects graph-structured dependencies to be be in enhanced UD format. The SemEval semantic dependencies datasets are in a different, very confusing and terrible format that will need to be converted. `scripts/to_sdp.py` will convert a .conllu file with graph-structured content back to the terrible .sdp format so you can evaluate it with the official evaluation script (written in java), but for convenience `scripts/semdep_eval.py` can be used to compute labeled F1 instead (I tested it against the official eval script and it reported identical LF1).

Probably, though, you don't just want to train the default model. At the very least the training data and word embeddings will be in a different location. One option is to specify your own configuration file and use that instead:
```bash
python3 main.py train ParserNetwork --config_file config/myconfig.cfg
```
`config/myconfig.cfg` can be underspecified, with all default parameters being copied over. To tell the model where to save the model, you can use the `--save_dir` flag. Note that this should go before `train`! This ordering forces some readability and makes the training/running syntax the same.
```bash
python3 main.py --save_dir saves/mymodel train ParserNetwork --config_file config/myconfig.cfg
```
Config options can be set on the command line as well. The syntax for doing this is `--SectionName option_1=value_1 option_2=value_2`.
```bash
python3 main.py --save_dir saves/mymodel train ParserNetwork --config_file config/myconfig.cfg --BaseNetwork recur_size=300 recur_keep_prob=.67 --FormTokenVocab min_occur_count=5
```

The config file contains some convenience variables in the DEFAULT section; these can be used when training a bunch of systems on different treebanks. For example, to loop through all the conll18 treebanks, point `save_metadir` to the place you want the models to be saved and point `data_metadir` to the place where the treebanks are located, and the following code will train a parser on each dataset.
```bash
cat conll18_treebanks.txt | while read -r LANG TREEBANK LC TB
do
  python main.py train ParserNetwork --config_file config/myconfig.cfg --DEFAULT LANG=$LANG TREEBANK=$TREEBANK LC=$LC TB=$TB
done
```
**NB** You also need to point the pretrained embeddings to the right place! They're located in the [FormPretrainedVocab] section. Reading them in from text is super slow, so if you want you can cache them somewhere as a pickle file to speed things up in subsequent calls. If `save_as_pickle` is True and `vocab_loadname` is specified then it'll try to load them from there first, and if it can't, it'll save them there after it reads them in the first time.

If you want to know all the configuration options, there's a ReadMe in the `config` directory that explains what each one does.

By default, if the save directory already exists, you'll get a prompt warning you that the system will delete it if you continue and giving you one last chance to opt-out. If this is annoying you (e.g. because you're debugging or something) you can skip this prompt with the `--force` flag. The system also produces a nifty training screen (using Python's `curses` package) that refreshes every 100 training steps, instead of spitting out thousands of lines to stdout. If your setup doesn't play nice with curses or if you want a textual record of the training history or if you want to debug with stdout print statements, you can disable this with `--noscreen`. You won't get the pretty colors though.
```bash
python main.py --save_dir saves/debug train ParserNetwork --config_file config/myconfig.cfg --force --noscreen
```

### Running
The model can be run by calling
```bash
python main.py --save_dir saves/mysave run files
```
There are a couple annoying quirks that might change though. Because of constraints during the CoNLL 2018 shared task, if there's only one file and no other flags are specified, it will print the file to standard out--but if there are multiple files, it will save them to `saves/mysave/parsed/$file`, where `$file` is the each original file, including any subdirectory structure. So if `$file` is `data/CoNLL18/UD_English-EWT/ud_ewt-en-dev.conllu`, then the output will be saved to `saves/mysave/parsed/data/CoNLL18/UD_English-EWT/ud_ewt-en-dev.conllu`. If you want to save the files somewhere else, you can use the `--output_dir` flag to redirect them to a different directory without copying the whole subdirectory structure.
```bash
python main.py --save_dir saves/mysave run --output_dir parses/ files
```
This will save everything to the `parses/` directory--make sure no files in different directories have the same basename though, or one will get overwritten!
If you have only one file and you want to change the basename, you can use the `--output_filename` flag.

**NB** In the future I might change it so that it always prints to standard out if you don't use any flags, but allow an `--output_dir` flag that saves the files to disk somewhere (in the `save_dir/parsed/` directory if unspecified), printing out the name of the saved file for convenience. Maybe with an `--ignore_subdirs` flag that ignores subdirectory structure when saving. I'll think about it.

### Optimizing
Proper hyperparameter tuning is key to getting peak performance, and honestly is just good science. Like if you have two models with different architectures, but one of them didn't tune their hyperparameters very well, how can you say which one is actually better? If you want to use an experimental hyperparameter tuning algorithm I've been playing with, there's some support for it in the code (see `main.py` and the `hpo` directory), but at the moment it's fairly ad-hoc. Just promise not to scoop me if you use it.

## Package structure
If you want to modify the source code for this directory, here's how it's structured:
* `main.py` The file that runs everything.
* `config/` The config files. Includes config files to replicate the hyperparameters of previous systems.
* `parser/` Where pretty much everything you care about is.
    * `config.py` Extends `SafeConfigParser`. Stores configuration settings.
    * `base_network.py` Contains architecture-neutral code for training networks.
    * `*_network.py` The tensorflow architecture for different kinds of networks.
    * `graph_outputs.py` An incomprehensible mish-mash of stuff. Designed to help printing performance/storing predictions/tracking history without making `base_network.py` or `conllu_dataset.py` too incomprehensible, but could probably be migrated to those two. Or at least cleaned up. I'm sorry.
    * `structs/` Contains data structures that store information.
        * `conllu_dataset.py` Used to manage a single dataset (e.g. trainset, devset, testset) or group of files.
        * `vocabs/` Stores the different kind of vocabularies.
            * `conllu_vocabs.py` Used for subclassing different vocabs so they know which column of the conllu file to read from
            * `base_vocabs.py` Methods and functions relevant to all vocabularies.
            * `index_vocabs.py` Used to store simple indices, such as the index of the token in the sentence of the index of the token's head. (Includes semantic dependency stuff)
            * `token_vocabs.py` Stores discrete, string-valued tokens (words, tags, dependency relations). (Includes semantic dependency stuff)
            * `subtoken_vocabs.py` Stores sequences of discrete subtokens (where the character-level things happen).
            * `pretrained_vocabs.py` For managing word2vec or glove embeddings.
            * `multivocabs.py` Used for managing networks that use more than one kind of word representation (token, subtoken, pretrained)
            * `feature_vocabs.py` For Universal Features and some types of composite POS tags.
        * `buckets/` Code for grouping sequences by length.
            * `dict_multibucket.py` For each vocab, contains a two-column array representing each sentence in the file (in order), where the first column points to a bucket and the second points to an index in the bucket
            * `dict_bucket.py` A bucket that contains all sequences of a certain length range for a given vocab.
            * `list_multibucket.py/list_bucket.py` The bucketing system for character vocabs. Same basic idea but not keyed by vocabulary.
    * `neural/` Where I dump tensorflow code.
        * `nn.py` Tensorflow is great but sometimes it's dumb and I need to fix it.
        * `nonlin.py` All the nonlinearities I might want to try stored in the same file.
        * `embeddings.py` Home-brewn TF functions for embeddings.
        * `recurrent.py` Tensorflow's built-in RNN classes aren't flexible enough for me so I wrote my own.
        * `classifiers.py` TF functions for doing linear/bilinear/attentive transformations.
        * `optimizers.py/` My own handcrafted optimizers. Builds in gradient clipping and doesn't update weights with `nan` gradients. 
            * `optimizer.py` The basic stuff. I'll probably rename it to `base_optimizer.py` for consistency.
            * `adam_optimizer.py` Implements Adam. Renames beta1 and beta2 to mu and nu, and adds in a gamma hp to allow for Nesterov momentum (which probably doesn't actually do anything)
            * `amsgrad_optimizer.py` Implements the AMSGrad fix for Adam. Unlike Adam, AMSGrad struggles when the network is poorly initialized--so if you want to use this in the code you have to set `switch_optimizers` to `True`. This will kick off training with Adam and then switch to AMSGrad after the network has gotten to a point where the gradients aren't so large they break it.
* `scripts/` Where I dump random scripts I write and also Chu-Liu/Edmonds.
* `debug/` Contains a simple little debugging tool that computes how much time and memory a block of code took.
* `hpo/` Some hyperparameter tuning algorithms.
