# ----------------------------------------------------------------------
# Parameters

FEAT = bert
BERT = mbert
CONFIG = config.ini

GPU = 0

#BUCKETS = --buckets=48
#BATCH_SIZE = --batch-size=500
#MAX_SENT_LENGTH=--max-sent-length 140
#ATTN=--attention-layer=6

#----------------------------------------------------------------------
# Corpora

CORPUS_DIR = ..
CORPUS_TRAIN = $(CORPUS_DIR)/train-dev/UD_$(RES2)/$(CORPUS)-ud-train.conllu
CORPUS_DEV = $(CORPUS_DIR)/train-dev/UD_$(RES2)/$(CORPUS)-ud-dev.conllu

#BLIND_TEST=$(CORPUS_DIR)/test-udpipe/$(LANG).conllu
#BLIND_TEST=$(CORPUS_DIR)/test-stanza-sent/$(LANG).conllu
#BLIND_TEST=$(CORPUS_DIR)/EDparser/data/iwpt2020/test-udpipe/$(LANG).conllu
BLIND_TEST=$(CORPUS_DIR)/iwpt2020stdata/sysoutputs/turkunlp/primary/$(LANG).conllu

GOLD_TEST= $(CORPUS_DIR)/iwpt2020stdata/$(UD_TOOLS)/../test-gold/$(LANG).conllu

UD_TOOLS = $(CORPUS_DIR)/iwpt2020stdata/tools

ifeq ($(LANG), ar)
  CORPUS=ar_padt
  RES2=Arabic-PADT
  MODEL = --bert=asafaya/bert-large-arabic #TurkuNLP/wikibert-base-ar-cased
  BERT = asafaya
else ifeq ($(LANG), ba)
  CORPUS=ba
  RES2=Baltic
  #MODEL = --bert=TurkuNLP/wikibert-base-lv-cased
lse ifeq ($(LANG), bg)
  CORPUS=bg_btb
  RES2=Bulgarian-BTB
  MODEL = --bert=DeepPavlov/bert-base-bg-cs-pl-ru-cased #TurkuNLP/wikibert-base-bg-cased #iarfmoose/roberta-base-bulgarian
  BERT = DeepPavlov
else ifeq ($(LANG), cs) #dev PDT
  CORPUS=cs_pdt
  RES2=Czech-PDT
  MODEL = --bert=DeepPavlov/bert-base-bg-cs-pl-ru-cased
  BERT = DeepPavlov
else ifeq ($(LANG), en)
  CORPUS=en_ewt
  RES2=English-EWT
  MODEL = --bert=google/electra-base-discriminator
  BERT = electra-base
else ifeq ($(LANG), ptb)
  CORPUS=en_ptb
  CORPUS_TRAIN = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-train.conllu
  CORPUS_DEV = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-dev.conllu
  BLIND_TEST = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-test.conllu
  GOLD_TEST = $(CORPUS_DIR)/SD_English_PTB/en_ptb-sd-test.conllu
  MODEL = --bert=google/electra-base-discriminator
  BERT = electra-base
else ifeq ($(LANG), et) #dev EDT
  CORPUS=et
  RES2=Estonian
  #MODEL = --bert=TurkuNLP/wikibert-base-et-cased
else ifeq ($(LANG), fi)
  CORPUS=fi_tdt
  RES2=Finnish-TDT
  MODEL = --bert=TurkuNLP/bert-base-finnish-cased-v1
  #MODEL = --bert=TurkuNLP/wikibert-base-fi-cased
  BERT = TurkuNLP
  #ATTN=--attention-layer=8
else ifeq ($(LANG), fr)
  CORPUS=fr_sequoia
  RES2=French-Sequoia
  MODEL = --bert=camembert/camembert-large #camembert-base TurkuNLP/wikibert-base-fr-cased
  BERT = camembert-large
else ifeq ($(LANG), it)
  CORPUS=it_isdt
  RES2=Italian-ISDT
  MODEL = --bert= dbmdz/bert-base-italian-xxl-cased
  BERT = dbmdz-xxl
else ifeq ($(LANG), lt)
  CORPUS=lt_alksnis
  RES2=Lithuanian-ALKSNIS
  #MODEL = --bert=TurkuNLP/wikibert-base-lt-cased
  ATTN=--attention-layer=8
else ifeq ($(LANG), lv)
  CORPUS=lv_lvtb
  RES2=Latvian-LVTB
  #MODEL = --bert=TurkuNLP/wikibert-base-lv-cased
else ifeq ($(LANG), nl) #dev Alpino
  CORPUS=nl
  RES2=Dutch
  #MODEL = --bert=TurkuNLP/wikibert-base-nl-cased
  MODEL = --bert=wietsedv/bert-base-dutch-cased
  BERT = wietsedv
else ifeq ($(LANG), pl) #dev LFG
  CORPUS=pl
  RES2=Polish
  MODEL = --bert=dkleczek/bert-base-polish-cased-v1 #DeepPavlov/bert-base-bg-cs-pl-ru-cased
  BERT = dkleczek
else ifeq ($(LANG), ru)
  CORPUS=ru_syntagrus
  RES2=Russian-SynTagRus
  MODEL = --bert=DeepPavlov/rubert-base-cased
  BERT = DeepPavlov
else ifeq ($(LANG), sk)
  CORPUS=sk_snk
  RES2=Slovak-SNK
  #MODEL = --bert=TurkuNLP/wikibert-base-sk-cased
else ifeq ($(LANG), sv)
  CORPUS=sv_talbanken
  RES2=Swedish-Talbanken
  MODEL = --bert=KB/bert-base-swedish-cased
  BERT = KB
else ifeq ($(LANG), ta)
  CORPUS=ta_ttb
  RES2=Tamil-TTB
  BLIND_TEST = $(CORPUS_DIR)/test-udpipe/$(LANG).conllu
  #MODEL = --bert=monsoon-nlp/tamillion
else ifeq ($(LANG), uk)
  CORPUS=uk_iu
  RES2=Ukrainian-IU
  MODEL = --bert=TurkuNLP/wikibert-base-uk-cased
  BERT = TurkuNLP
  # nu=0.9
else ifeq ($(LANG), zh)
  CORPUS=zh
  CORPUS_TRAIN = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-train.conllu
  CORPUS_DEV = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-dev.conllu
  BLIND_TEST = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-test.conllu
  GOLD_TEST = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-test.conllu
  MODEL = --bert=hfl/chinese-electra-base-discriminator # bert-base-chinese # hfl/chinese-electra-large-discriminator
  BERT = hfl
else
  CORPUS_TRAIN= data/CoNLL2009-ST-English-train.conll
  CORPUS_DEV  = data/CoNLL2009-ST-English-development.conll
  CORPUS_TEST = data/CoNLL2009-ST-English-test-wsj.conll
endif

#----------------------------------------------------------------------
# Targets

.PRECIOUS: exp/$(LANG)-$(FEAT)$(VER)/model

exp/$(LANG)-$(FEAT)$(VER)/model:
	python -u -m parser.cmds.biaffine_dependency train -d=$(GPU) -p=$@ \
	   -c=$(CONFIG) $(MODEL) $(ATTN) \
	   --train=$(CORPUS_TRAIN) $(MAX_SENT_LENGTH) $(BATCH_SIZE) $(BUCKETS) \
	   --dev=$(CORPUS_DEV) --feat=$(FEAT)

exp/$(LANG)-$(FEAT)$(VER)-test.conllu: exp/$(LANG)-$(FEAT)$(VER)/model
	python -m parser.cmds.biaffine_dependency predict -d=$(GPU) -p=$< --tree \
	   --data=$(BLIND_TEST) \
	   --pred=$@
	python $(CORPUS_DIR)/fix-root.py $@

exp/$(LANG)-$(FEAT)$(VER)-test.time: exp/$(LANG)-$(FEAT)$(VER)/model
	( time python -m parser.cmds.biaffine_dependency predict -d=$(GPU) -p=$< --feat=$(FEAT) --tree  \
	   $(BLIND_TEST)  \
	   --pred=/dev/null; ) &> $@

LANGS=ar bg cs en et fi fr it lt lv nl pl ru sk sv ta uk 
LANGS1=ar bg en et fi sk
LANGS2=fr it ru ta uk sv
LANGS3=lv lt nl pl cs

all:
	for l in $(LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}-$(FEAT)$(VER)-test.eval &>> exp/$${l}-$(FEAT)$(VER)-test.make; \
	done

train:
	for l in $(LANGS); do \
	    nohup ${MAKE} -s GPU=$(GPU) LANG=$$l exp/$$l-$(FEAT)$(VER)/model &>> exp/$${l}-$(FEAT)$(VER).make; \
	done

# ----------------------------------------------------------------------
# Evaluation

%-test.nen.conllu: %-test.conllu
	   perl $(UD_TOOLS)/enhanced_collapse_empty_nodes.pl $< > $@

%-test.eval: %-test.nen.conllu
	python $(UD_TOOLS)/iwpt20_xud_eval.py -v $(UD_TOOLS)/../test-gold/$(LANG).nen.conllu $< > $@

%-test.evalb: %-test.eval
	python $(CORPUS_DIR)/eval.py -g $(GOLD_TEST) -s $@ --evalb

%-test.eval07: %-test.conllu
	perl $(CORPUS_DIR)/eval07.pl -p -q -g $(GOLD_TEST) -s $< > $@

evaluate:
	for l in $(LANGS); do \
	   $(MAKE) -s GPU=$(GPU) LANG=$$l exp/$$l-$(FEAT)-test.evalb &>> exp/$$l-$(FEAT)-test.make; \
	done

exp/test.eval: evaluate
	( cd exp; python ../eval-summary.py > $(notdir $@) )

baltic:
	for l in et lt lv; do \
	  python -m parser.cmds.biaffine_dependency predict -d=$(GPU) --feat=$(FEAT) \
	   -p=exp/ba-$(FEAT) --tree \
	   $(subst $(LANG),$$l,$(BLIND_TEST)) \
	   --pred=exp/$$l-$(FEAT)-ba-test.conllu; \
	done

# ----------------------------------------------------------------------
# Run tests

test:
	pytest -s tests

# ----------------------------------------------------------------------
# Parse plain text

TEXT_FILE=

# example
# make GPU=2 LAN=it CORPUS_DIR=/project/piqasso/Collection/IWPT20 TEXT_FILE=/project/piqasso/Collection/IWPT20/train-dev/UD_Italian-ISDT/it_isdt-ud-dev.txt exp/it-bert-raw-text.conllu
exp/$(LAN)-$(FEAT)$(VER)-$(TEXT_FILE).conllu: exp/$(LAN)-$(FEAT)$(VER)/model
	python -m parser.cmds.biaffine_dependency predict -d=$(GPU) -p=$< --tree --text $(LAN) \
	   $(TEXT_FILE) \
	   --pred=$@
