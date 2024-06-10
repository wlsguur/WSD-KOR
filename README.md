# Korean Word Sense Disambiguation via Prompt-based Learning

## 프롬프트 학습을 이용한 한국어 단어 중의성 해소 프로젝트

COSE461-02 Natural Language Processing

## 개요

```
* '형'은 자상하고 애정이 깊었으며 언제나 너그러웠다.
* 그는 법정에서 12년 '형'을 선고 받았다.
```

위 두 문장에서 공통적으로 사용된 단어 '형'은 여러가지 의미를 가지는 <b>다의어</b>입니다. 첫 번째 문장에서 '형'은 '같은 부모에게서 태어난 남자 중 손윗사람'라는 의미로, 두 번째 '형'은 '범죄에 대한 법적 제재'이라는 의미로 사용되었습니다. 이처럼 다의어가 주어진 문장에서 어떤 의미로 사용되었는지를 분석하는 것을 단어 중의성 해소(Word Sense Disambiguation, 이하 WSD)라고 합니다. 한국어 WSD를 위한 딥러닝 모델을 구현하는 것이 목표입니다.

해당 모델의 구현을 위한 데이터셋은 <b>국립국어원</b>에서 제공하는 [<b>어휘 의미 분석 말뭉치</b>](https://corpus.korean.go.kr/)를 이용하였습니다. 이 말뭉치에서는 [<b>우리말샘</b>](https://opendict.korean.go.kr/main) 사전의 의미들을 기준으로, 주어진 문장에서 다의어 <b>명사</b>들이 가지는 의미를 정리하였습니다. 따라서 이번 분석에서 다의어는 <b>우리말샘 사전 기준 의미가 2개 이상인 명사</b>로 한정하였습니다.

문장의 다의어를 분석하기 위해 사용한 모델은 [KoELECTRA](https://github.com/monologg/KoELECTRA.git)입니다. <b>KoELECTRA</b>는 [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB) 라는 논문을 통해 공개된 <b>ELECTRA</b> 모델을 한국어로 학습시킨 모델입니다. 사전 학습된 KoELECTRA를 기반으로 [Pytorch](https://pytorch.org/)와 [Huggingface Transformers](https://github.com/huggingface/transformers)를 사용하여 해당 모델을 구현하였습니다.

모델 학습에는 <b>prompt-based learning</b> 방법을 이용하였습니다. 이는 [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)라는 논문에서 제안한 방법으로, 기존 입력 $x$에 prompting function $f$를 적용한 $x' = f(x)$를 모델의 입력으로 하여 학습시키는 방법입니다.

## 어휘 의미 분석 말뭉치 

어휘 의미 분석 말뭉치는 문어와 구어로 이루어져 있으며, '문맥/문장/단어/다의어 의미 정보'의 딕셔너리로 구축되어 있습니다. 어휘 의미 분석 말뭉치의 통계 정보는 다음과 같습니다. (각 문어, 구어, 합계 순)

* 문맥 : 7265 + 423 = 7688개
* 문장 : 150,082 + 223,962 = 374,044개
* 어절 : 2,000,213 + 1,006,447 = 3,006,660개
* 다의어 : 1,672,688 + 467,480 = 2,140,168개

이 데이터셋은 문맥/문장 단위로 다음과 같은 JSON 형태로 주어집니다.

```
{
 'id': 'SARW1800000001.1',
 'form': '요즘처럼 추운 날씨에는',
 'word': [{'begin': 0, 'end': 4, 'form': '요즘처럼', 'id': 1},
          {'begin': 5, 'end': 7, 'form': '추운', 'id': 2},
          {'begin': 8, 'end': 12, 'form': '날씨에는', 'id': 3}],
 'morpheme': [{'form': '요즘',
               'id': 1,
               'label': 'NNG',
               'position': 1,
               'word_id': 1},
              {'form': '처럼',
               'id': 2,
               'label': 'JKB',
               'position': 2,
               'word_id': 1},
              {'form': '춥',
               'id': 3,
               'label': 'VA',
               'position': 1,
               'word_id': 2},
              {'form': 'ㄴ',
               'id': 4,
               'label': 'ETM',
               'position': 2,
               'word_id': 2},
              {'form': '날씨',
               'id': 5,
               'label': 'NNG',
               'position': 1,
               'word_id': 3},
              {'form': '에',
               'id': 6,
               'label': 'JKB',
               'position': 2,
               'word_id': 3},
              {'form': '는',
               'id': 7,
               'label': 'JX',
               'position': 3,
               'word_id': 3}],
 'WSD': [{'begin': 0,
          'end': 2,
          'pos': 'NNG',
          'sense_id': 1,
          'word': '요즘',
          'word_id': 1},
         {'begin': 8,
          'end': 10,
          'pos': 'NNG',
          'sense_id': 1,
          'word': '날씨',
          'word_id': 3}],          
}
```
각 key에 대응되는 정보는 다음과 같습니다.

* id : 문장별 고유번호
* form : 문장
* word : 문장을 구성하는 어절
* morpheme : 문장을 구성하는 형태소
* WSD : 문장의 다의어 

다의어의 의미에 대한 정보는 `WSD`라는 키에 대응되어 있으며, 이를 통해 문장에서 해당 다의어의 인덱스 정보(`begin`, `end`), 형태소 정보(`pos`), 우리말샘 기준 의미 정보(`sense_id`) 등의 정보를 제공합니다. 

어휘 의미 말뭉치는 국립국어원 말뭉치 홈페이지에서 신청서를 작성, 허가 후 다운로드 받을 수 있습니다. 더 자세한 사항은 해당 데이터를 구축한 고려대학교 연구진이 국립국어원에 제출한 [분석보고서](https://korean.go.kr/common/download.do;front=705CF43F5B77029E1B5BE09E8910830F?file_path=reportData&c_file_name=f7222492-4580-40c6-864f-b66caeeeab3c_0.pdf&o_file_name=%EC%B5%9C%EC%A2%85%20%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%96%B4%ED%9C%98%EC%9D%98%EB%AF%B8%20%EB%B6%84%EC%84%9D%20%EB%A7%90%EB%AD%89%EC%B9%98%20%EA%B5%AC%EC%B6%95.pdf)에 수록되어 있습니다.

이 모델에서는 어휘 말뭉치 중 매 10번째 문맥 데이터를 평가 데이터, 그 외를 훈련 데이터로 사용하였습니다.


## 우리말샘 사전

다음은 우리말샘 사전 [홈페이지](https://opendict.korean.go.kr/main)에서 발췌한 우리말샘 소개글입니다.

> 우리말의 쓰임이 궁금할 때 국어사전을 찾게 됩니다. 그런데 막상 사전을 찾아도 정보가 없거나 설명이 어려워 아쉬움을 느낄 때가 있습니다. 그동안 간행된 사전들은 여러 가지 제약이 있어 정보를 압축하여 제한적으로 수록하였기 때문입니다.<br> 사용자가 참여하는 ‘우리말샘’은 이런 문제점을 극복하고자 기획되었습니다. 한국어를 사용하는 우리 모두가 주체가 되어 예전에 사용되었거나 현재 사용되고 있는 어휘를 더욱 다양하고 알기 쉽게 수록하고자 합니다. 또한 전통적인 사전 정보 이외에 다양한 언어 지식도 실어 한국어에 관한 많은 궁금증을 푸는 통로가 되고자 합니다.

우리말샘은 다음과 같이 전문가가 감수한 전통적인 의미에 더해 참여자가 제안한 정보 또한 수록되어, 갈수록 다양해지고 있는 어휘의 의미들을 품기에 적합한 사전입니다. 누구나 자유롭게 이용할 수 있는 `크리에이티브 커먼즈 저작자표시-동일조건변경허락 2.0 대한민국 라이선스`에 따라 배포되며, 회원 가입 후 사전 전체를 XML 파일로 다운로드 할 수 있습니다. 


## Prompt-based Learning

![prompt-based learning](https://github.com/wlsguur/WSD_KOR/assets/140404752/4fac8679-5675-4120-825c-d87e61cea105)

Prompt-based laerning은 기존 입력 $x$에 prompting function $f$를 적용한 $x' = f(x)$를 모델의 입력으로 하여 fine-tuning 시키는 방법입니다. 그림과 같이 감정 분석을 수행하는 경우, 입력의 형태를 [MASK] 토큰을 포함한 프롬프트 형식으로 바꾸어 모델이 토큰에 해당하는 단어를 학습하도록 합니다. 모델이 예측한 단어를 다시 기존 label에 메핑시켜 loss를 흘려주어 모델을 학습시킵니다.

## 모델 학습 결과 

학습에는 [KoELECTRA-Base-v3](https://huggingface.co/monologg/koelectra-base-v3-discriminator) 을 이용

| Model      	| Accuracy 	| F1(weighted) 	|
|------------	|----------	|--------------	|
| [Bi-Encoder (DistilKoBERT)] (https://github.com/lih0905/WSD_kor.git)  	| 88.33  	| 0.883      	|
| [KoELECTRA](https://www-dbpia-co-kr-ssl.oca.korea.ac.kr/journal/articleDetail?nodeId=NODE11224131) 	| 92.90  	| -       |
| [KoELECTRA + prompt based learning](https://www-dbpia-co-kr-ssl.oca.korea.ac.kr/journal/articleDetail?nodeId=NODE11224131)	| 93.70  	| -       |
| KoELECTRA + CLS head + prompt based laerning (ours) 	| <b>96.52</b>  	| <b>0.965</b>       |

## 사용법

* 필요 라이브러리
    * Python >= 3.7
    * Pytorch >= 1.6.0
    * Huggingface Transformers >= 3.3.0
    * Pandas
    * scikit-learn
    * tqdm
    * wandb (optional)

* 깃 리포지토리 복사

    ```bash
    git clone https://github.com/wlsguur/WSD_KOR.git
    ```

* 필요 라이브러리 설치

    ```bash
    pip install -r requirments.txt
    ```

* 전처리
    * 국립국어원 말뭉치 데이터를 다운로드 후 `Data` 폴더에 저장합니다.
    * 우리말샘 사전 데이터를 XML 형식으로 다운받은 후 `Dict` 폴더에 저장합니다.

    ```bash
    python preprocess.py
    ```

* 프롬프트를 적용한 데이터셋 생성
    * `prompt.py`에서 프롬프트 형식을 변경할 수 있습니다.

    ```python
    def prompt_generator(contexts, dictionary, use_all=False, split=False, test_size = 0.3):
        '''
        ~
        '''
        for i in range(len(contexts)):
            context = contexts['form'][i]
            line = contexts['WSD'][i]
            for d in line:
                target_wsd, target_sense_id = d['word'], d['sense_id']
                if target_sense_id not in dictionary[target_wsd]["sense_no"]:
                    continue
                if len(dictionary[target_wsd]["definition"]) > 3:
                    idx = dictionary[target_wsd]["sense_no"].index(target_sense_id)
                    if idx >= 3:
                        idxs = [0, 1, idx]
                    else:
                        idxs = range(3)
                else:
                    idxs = range(len(dictionary[target_wsd]["definition"]))
                for j in idxs:
                    candidate_definition = dictionary[target_wsd]["definition"][j]
                    candidate_sense_id = dictionary[target_wsd]["sense_no"][j]
                    # 프롬프트 생성
                    input = (
                        f"주어진 문장을 보고, 단어의 의미를 묻는 질문에 문맥을 고려하여 답하세요.\n"
                        f"문장: {context}\n"
                        f"문장에서의 \"{target_wsd}\"은(는) \"{candidate_definition}\"의 의미로 쓰였다.\n"
                    )
                    inputs.append(input)
                    labels.append(1 if candidate_sense_id == target_sense_id else 0)
    
        return {"inputs": inputs, "labels": labels}
    ```

* Baseline 모델 검증

    ```bash
    python baseline.py
    ```

* KoELECTRA 모델 훈련 및 검증

    ```bash
    python train.py
    ```

## References

* [국립국어원(2020). 국립국어원 어휘 의미 분석 말뭉치(버전 1.0)](https://corpus.korean.go.kr/)
* [KoELECTRA](https://github.com/monologg/KoELECTRA.git)
* [한국어 어휘 의미 분석 모델](https://github.com/lih0905/WSD_kor.git)
* [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)
* [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/pdf/2107.13586)
* [민진우, 나승훈. (2022-12-20). 프롬프트 학습 기반 한국어 개체 중의성 해결. 한국정보과학회 학술발표논문집, 제주.](https://www-dbpia-co-kr-ssl.oca.korea.ac.kr/journal/articleDetail?nodeId=NODE11224131)
* [NLP-progress : Word Sense Disambiguation](http://nlpprogress.com/english/word_sense_disambiguation.html)
