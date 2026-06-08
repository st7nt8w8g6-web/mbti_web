# ============================================================
# 문체 MBTI 분류기  (style_mbti_v2.py)
# ============================================================
# 코랩 실행 전 준비 (셀 1):
#   !apt-get install -y fonts-nanum
#   !pip install numpy matplotlib
#   import matplotlib.font_manager as fm
#   fm._load_fontmanager(try_read_cache=False)
# ============================================================
#
# [v2 변경 사항]
#  1. 축4(유보형-단정형): '~음/~임/~ㅁ'체 어미 탐지 → 단정형(J) 가산
#  2. 축2(서사형-압축형): 단어 수 기준 상향 (긴 문장 → S, 짧은 문장 → C)
#  3. 딕셔너리 누락/반영 오류 어휘 보정 및 비율 재조정
# ============================================================

import re, os, subprocess, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================
# 0-A. 한글 폰트 전역 설정
# ============================================================
def _setup_korean_font():
    nanum_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]
    for p in nanum_paths:
        if os.path.exists(p):
            fe = fm.FontEntry(fname=p, name="KorFont")
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams["font.family"] = "KorFont"
            plt.rcParams["axes.unicode_minus"] = False
            return fm.FontProperties(fname=p)

    for f in fm.fontManager.ttflist:
        comb = (f.name + f.fname).lower()
        if "notosanscjk" in comb or "noto sans cjk" in comb:
            if any(k in f.fname.lower() for k in ["regular", "medium"]):
                fe = fm.FontEntry(fname=f.fname, name="KorFont")
                fm.fontManager.ttflist.insert(0, fe)
                plt.rcParams["font.family"] = "KorFont"
                plt.rcParams["axes.unicode_minus"] = False
                return fm.FontProperties(fname=f.fname)

    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko", "--format=%{file}\n"],
            capture_output=True, text=True
        )
        ko_fonts = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        for f in ko_fonts:
            if any(k in f for k in ["Regular", "Medium", "Bold"]):
                fe = fm.FontEntry(fname=f, name="KorFont")
                fm.fontManager.ttflist.insert(0, fe)
                plt.rcParams["font.family"] = "KorFont"
                plt.rcParams["axes.unicode_minus"] = False
                return fm.FontProperties(fname=f)
    except Exception:
        pass

    plt.rcParams["axes.unicode_minus"] = False
    return fm.FontProperties()

FP = _setup_korean_font()


# ============================================================
# 0-B. 딕셔너리 정의
# ============================================================

# ── 축 1: 감성형(E) vs 논리형(L) ──────────────────────────
EMOTION_KEYWORDS = [
    # 기쁨·긍정
    "행복", "기쁘", "기뻤", "기쁜", "즐겁", "즐거", "반갑",
    "뿌듯", "흐뭇", "유쾌", "상쾌", "만족", "감격",
    "감동", "벅차", "환희", "기대", "설레", "설렘",
    "희열", "황홀", "행운", "축복", "감사", "고맙",
    "다행", "안도", "든든", "평온", "편안",
    # 슬픔
    "슬프", "슬펐", "슬픈", "서럽", "눈물", "울컥",
    "비통", "애달프", "비애", "상실", "공허", "허무",
    "쓸쓸", "처량", "고독", "외롭", "외로웠",
    "그립", "그리움", "먹먹", "침울", "우울",
    "비참", "허전", "안타깝",
    # 분노
    "화나", "분노", "격분", "짜증", "답답",
    "억울", "원망", "분개", "불쾌", "격앙",
    "격노", "증오", "미움", "혐오", "불만",
    "분통", "성나", "울분",
    # 불안·공포
    "불안", "걱정", "두렵", "두려움", "무섭",
    "무서", "초조", "긴장", "공포", "위태",
    "불확실", "조마조마", "떨리", "당황",
    "혼란", "위기감", "염려", "우려",
    # 사랑·애정
    "사랑", "애정", "정", "애틋",
    "소중", "아끼", "보고싶", "애정어린", "친밀", "정겨",
    "다정", "따뜻", "포근", "애착",
    # 공감·위로
    "위로", "공감", "이해", "배려",
    "격려", "응원", "위안", "돌봄",
    "보듬", "토닥", "안쓰럽", "가엾",
    "측은", "연민",
    # 희망·기대
    "희망", "소망", "꿈",
    "희망적", "낙관", "긍정", "바라",
    "염원", "확신", "용기", "의욕",
    "열정", "도전", "각오",
    # 후회·죄책감
    "후회", "죄책감", "미안", "아쉽",
    "유감", "반성", "부끄럽",
    "자책", "괴롭", "괴로웠",
    # 정서 상태
    "감정", "마음", "느낌", "기분",
    "심정", "정서", "감성", "심리",
    "속마음", "내면", "진심", "진정성",
    # 관계 감정
    "신뢰", "믿음", "존경", "경외",
    "친근", "유대", "연대", "정겨움",
    "친밀감", "동질감",
    # 문학·에세이에서 자주 등장
    "사무치", "절실", "간절",
    "애잔", "울림", "여운", "감회",
    "회한", "그리워", "애수", "정취",
    "낭만", "서정", "정감", "향수",
    # 감탄·정서적 평가
    "놀랍", "감탄", "경이", "신기",
    "경탄", "감명", "인상적", "경악",
    "충격", "당혹", "어이없",
    # 돌봄·인간관계
    "관심", "사랑스러",
    "고마움", "따스", "온정",
    "인정", "친절", "자상", "보살핌",
    # 1인칭
    "나는", "내가", "내게", "나를",
    "저는", "제가", "저를",
    "우리는", "우리", "우리의",
]

EVAL_ADVERBS = [
    "너무", "정말", "진짜", "엄청", "굉장히",
    "매우", "아주", "진심", "몹시",
]

LOGICAL_KEYWORDS = [
    # 인과
    "따라서", "그러므로", "결과적으로",
    "때문에", "이로 인해", "이에 따라",
    "원인", "결과", "영향", "효과",
    "기인", "초래", "유발", "발생",
    # 대조·반박
    "반면", "그러나", "하지만", "반대로",
    "한편", "다만", "반박", "대조",
    "상반", "대립", "차이", "대비",
    "반증", "예외",
    # 논증 연결
    "즉", "요컨대",
    "정리하면", "종합하면", "요약하면",
    "결론적으로", "종합적으로",
    "결국", "이를 통해",
    # 근거
    "근거", "증거", "자료", "사례",
    "통계", "수치", "실험", "조사",
    "관찰", "검증", "입증", "확인",
    "실증", "데이터", "증명",
    # 추론
    "추론", "가정", "가설", "예측",
    "추정", "판단", "논리", "추측",
    "도출", "귀납", "연역",
    # 분석
    "분석", "해석", "설명", "검토",
    "고찰", "평가", "비교", "논의",
    "탐구", "연구", "분류", "정리",
    # 구조
    "구조", "체계", "메커니즘", "원리",
    "과정", "단계", "유형", "범주",
    "개념", "맥락", "관계", "모형",
    "프레임", "체계성",
    # 학술담화
    "현상", "본질", "특징", "속성",
    "성격", "양상", "변화", "경향",
    "패턴", "기능", "역할",
    # 열거
    "첫째", "둘째", "셋째", "넷째",
    "우선", "다음으로", "마지막으로",
    # 비교
    "비교", "공통점",
    "차이점", "유사", "상이",
    "구분",
    # 논문체
    "시사점", "함의",
    "결과물", "연구결과",
    "선행연구", "후속연구",
    "분석결과", "연구문제",
    "연구방법", "연구대상",
    # 사회과학
    "변수", "요인", "상관관계",
    "인과관계", "독립변수",
    "종속변수", "매개", "조절",
    # 인문학
    "담론", "텍스트",
    "재현", "의미", "해석학",
    "서사", "관점", "관념",
    # 컴퓨터·데이터
    "알고리즘", "모델", "최적화",
    "분류기",
    # 평가
    "타당성", "신뢰성",
    "객관성", "일관성",
    "적절성", "유효성",
    # 결론
    "결론", "요약",
    "종합", "귀결", "함축",
    "란", "이란",
]

LOGICAL_MULTI = [
    "예를 들어", "다시 말해", "핵심은", "결론적으로",
    "분석하면", "구조적으로", "그 결과", "를 의미한다",
    "를 뜻한다", "정의된다", "라고 정의한다",
]

# ── 축 2: 서사형(S) vs 압축형(C) ──────────────────────────
TEMPORAL_KEYWORDS = [
    # 시간 지시
    "그때", "당시", "예전", "옛날",
    "어릴적", "어릴때", "한때",
    "과거", "현재", "이후", "그후",
    "그전", "그전에", "그뒤", "그뒤로",
    # 사건 전환
    "그러다가", "그러던중", "그러던",
    "그러면서", "그러자", "그러고",
    "이어", "이어서", "뒤이어",
    "갑자기", "문득", "우연히",
    # 과정
    "계속", "점점", "차츰",
    "서서히", "점차", "점진적",
    "갈수록", "차차", "그래서", "그 다음부터",
    # 시작
    "처음", "처음에는", "처음엔", "처음부터",
    "시작", "출발", "이제",
    # 중간
    "도중", "중간에", "한동안", "한참", "그무렵",
    # 회상
    "기억난다", "기억난",
    "떠오른다", "회상",
    "돌이켜보면", "생각해보면",
    "돌아보면", "되돌아보면",
    # 경험 동사 (서사형의 핵심 — 사전에서 직접 매칭)
    "겪었다", "겪은",
    "보았다", "봤다",
    "느꼈다", "알게되었다",
    "깨달았다", "경험했다",
    "만났다",
    # 빈도
    "언제나", "항상", "자주", "종종", "가끔", "때때로",
    # 특정 시점
    "어느날", "어느순간", "그순간", "그시점", "그날", "그날밤",
]

TEMPORAL_MULTI = [
    "그 후", "잠시 후", "얼마 뒤", "얼마 후", "한참 뒤",
    "그러고 나서", "그러던 중", "그러던 어느 날",
    "시간이 지나", "시간이 흐른 뒤", "세월이 흐르면서",
    "처음에는", "처음엔", "처음부터",
    "나중에는", "나중엔", "결국에는",
    "그 순간", "바로 그 순간",
    "어느 날", "어느 시점에",
    "한편", "그 무렵",
    "뒤이어", "이어서",
    "그 일을 계기로", "그 사건 이후",
    "지나고 보니", "돌이켜 생각해보면",
    "시간이 갈수록", "세월이 지날수록",
    "겪었다", "보았다", "느꼈다",
    "만났다", "알게되었다", "깨달았다", "경험했다",
]

SUMMARY_KEYWORDS = [
    # 핵심
    "핵심", "본질", "요점", "관건", "중점", "주안점",
    # 결론
    "결론", "귀결", "결국", "따라서", "그러므로",
    # 요약
    "요약", "정리", "종합", "압축",
    # 일반화
    "전반적", "전체적", "종합적", "포괄적",
    # 논증
    "즉", "요컨대", "다시말해",
    # 구조
    "첫째", "둘째", "셋째", "넷째",
    # 평가
    "중요", "의의", "시사점", "함의",
    # 학술
    "결과적으로", "종합적으로", "본질적으로", "핵심적으로",
    # 메타담화
    "요약", "종합",
]

SUMMARY_MULTI = [
    "요약하면", "정리하면", "정리하자면",
    "결론적으로", "결론부터 말하면",
    "다시 말해", "쉽게 말해", "간단히 말해",
    "한마디로", "요컨대",
    "종합하면", "종합적으로 보면",
    "핵심은", "중요한 점은", "주목할 점은",
    "본질적으로", "결국에는",
    "이를 정리하면", "이를 요약하면",
    "전체적으로 보면", "전반적으로 보면",
    "핵심적으로는", "결론을 내리자면",
    "요점을 말하자면", "핵심만 말하면",
    "종합적으로 판단하면", "정리해서 말하면",
    "궁극적으로", "결과적으로 보면",
]

# ── 축 3: 직관형(I) vs 구체형(D) ──────────────────────────
ABSTRACT_NOUNS = [
    "존재", "실재", "본질", "실존",
    "자아", "의식", "무의식",
    "초월", "영원", "유한", "무한",
    "정체성", "주체", "타자",
    "관념", "이념", "사유", "철학",
    "인식", "인식론", "세계관",
    "가치", "윤리", "도덕",
    "진리", "정의", "자유",
    "평등", "책임", "의무",
    "의미", "상징", "개념", "범주",
    "원리", "원칙", "논리",
    "질서", "체계", "구조",
    "맥락", "관계", "현상",
    "본성", "속성", "특성",
    "제도", "문화", "사회",
    "권력", "이데올로기",
    "담론", "규범", "관습",
    "공동체", "정치", "경제",
    "서사", "텍스트", "담화",
    "재현", "해석", "의도",
    "관점", "시선", "주제",
    "상징성", "함의",
    "감정", "정서", "심리",
    "욕망", "동기", "충동",
    "기억", "정체감",
    "가설", "추론", "분석",
    "논증", "평가", "비판", "고찰",
    "가능성", "잠재력", "비전",
    "전망", "예측", "가정",
    "이상", "목표", "방향성",
    "패러다임", "프레임",
    "메커니즘", "모형",
    "시스템", "프로세스",
    "양상", "경향", "패턴",
    "시사점", "귀결",
    "전제", "명제", "조건",
    "추상성", "보편성",
    "특수성", "개별성",
    "상상력", "은유",
    "알레고리", "정조",
    "미학", "예술성",
    "상징체계",
]

SENSORY_KEYWORDS = [
    # 시각
    "눈", "시선", "빛", "햇빛", "햇살",
    "불빛", "어둠", "그림자", "광경",
    "풍경", "장면", "모습", "형태",
    "표정", "얼굴", "눈동자",
    "빨간", "붉은", "노란", "파란",
    "검은", "하얀", "푸른", "회색",
    "초록", "보라", "주황",
    "반짝", "희미", "선명", "환한",
    "어두운", "눈부신",
    # 청각
    "소리", "목소리", "울림",
    "비명", "속삭임", "함성",
    "고함", "대화", "노래",
    "음악", "멜로디",
    "웃음소리", "발소리",
    "빗소리", "바람소리",
    "물소리", "종소리", "천둥소리",
    # 후각
    "냄새", "향기", "향",
    "향수", "꽃향기",
    "비린내", "탄내",
    "흙냄새", "바다냄새",
    # 미각
    "맛", "단맛", "쓴맛",
    "신맛", "짠맛",
    "매운맛", "고소한",
    "달콤", "씁쓸",
    # 촉각·온도
    "손", "피부", "손끝",
    "촉감", "감촉",
    "차갑", "따뜻", "뜨겁",
    "미지근", "시리",
    "싸늘", "포근",
    "부드럽", "거칠",
    "매끈", "축축",
    "촉촉", "딱딱",
    # 신체
    "팔", "다리", "발",
    "어깨", "손가락",
    "입술", "머리카락",
    # 행동 동사 (구체형 핵심)
    "걷", "걸었", "뛰",
    "달리", "먹", "마시",
    "듣", "보다", "봤",
    "만지", "느끼",
    "움직", "앉", "일어",
    "누웠", "서있",
    "잡", "쥐", "안았",
    "바라", "쳐다",
    "둘러보", "들여다보",
    # 장소
    "방", "교실", "거리",
    "골목", "광장", "공원",
    "카페", "도서관",
    "학교", "지하철",
    "버스", "시장",
    "창문", "문", "복도",
    "계단", "베란다",
    # 자연
    "비", "눈", "바람",
    "구름", "하늘",
    "강", "호수",
    "산", "들판",
    "나무", "꽃",
    "잎", "흙",
    # 사물
    "책상", "의자",
    "컵", "유리잔",
    "가방", "책",
    "노트", "연필",
    "휴대폰", "컴퓨터",
    # 시간 (구체적 시점)
    "어제", "오늘",
    "내일", "아침",
    "점심", "저녁",
    "밤", "새벽",
    "오전", "오후",
]

SENSORY_MULTI = [
    "차갑다", "따뜻하다", "부드럽다",
    "보았다", "봤다", "들었다",
    "느꼈다", "만졌다", "잡았다",
    "먹었다", "마셨다",
    "걸었다", "뛰었다",
    "앉았다", "일어났다",
    "바라보았다", "쳐다보았다",
]

# ── 축 4: 유보형(F) vs 단정형(J) ──────────────────────────
HEDGE_KEYWORDS = [
    "아마", "어쩌면", "혹시",
    "왠지", "짐작컨대",
    "대체로", "대부분",
    "비교적", "상당히",
    "어느정도", "어느 정도",
    "부분적", "부분적으로",
    "불확실", "모호",
    "애매", "불분명",
    "명확하지", "확실하지",
    "가능", "가능성",
    "잠재적", "잠정적",
    "예비적",
    "다소", "약간",
    "조금", "일부",
    "일정부분",
    "만약", "경우에따라",
    "경우에 따라",
    "상황에 따라",
    "듯하다", "듯싶다",
    "추정", "예상",
    "짐작", "추측",
    "개인적으로",
    "주관적으로",
    "제 생각에는",
    "보기에",
    "일면", "한편으로는",
    "어쩌면", "반드시 그렇지는",
    "잠정적", "제한적",
    "가설적", "시범적",
    "탐색적",
]

HEDGE_MULTI = [
    "일 수 있다", "수도 있다", "할 수 있다", "될 수 있다",
    "가능성이 있다", "가능성이 높다", "가능성이 낮다",
    "것 같다", "것 같아", "듯하다", "듯 보인다",
    "보인다", "여겨진다", "생각된다", "판단된다",
    "추정된다", "예상된다", "짐작된다", "추측된다",
    "라고 볼 수 있다", "라고 해석할 수 있다", "라고 이해할 수 있다",
    "로 해석된다", "로 이해된다",
    "관점에서 보면", "라고 생각한다", "라고 생각해 볼 수 있다",
    "로 볼 여지가 있다", "라고 볼 여지가 있다",
    "경우에 따라", "상황에 따라", "조건에 따라", "맥락에 따라",
    "확실하지 않다", "명확하지 않다",
    "단정하기 어렵다", "판단하기 어렵다",
    "결론내리기 어렵다", "일반화하기 어렵다",
    "경향이 있다", "특징을 보인다", "나타나는 경향이 있다",
    "일 가능성이 있다", "일 가능성이 높다", "일 가능성이 낮다",
    "일 것으로 보인다", "일 것으로 판단된다",
    "시사하는 바가 있다", "암시하는 것으로 보인다",
    "추론할 수 있다", "간주될 수 있다",
    "해석될 수 있다", "설명될 수 있다",
    "이해될 수 있다", "논의될 수 있다",
    "한계가 있다", "제약이 있다",
    "제한적으로 해석해야 한다",
    "반드시 그렇지는 않다", "예외가 존재할 수 있다",
    "다른 해석도 가능하다",
    "일정 부분 설명할 수 있다",
    "탐색적으로 볼 때", "잠정적으로 볼 때",
    "가설적", "추론하면", "해석하면",
]

ASSERTIVE_KEYWORDS = [
    "반드시", "분명히", "확실히", "명확히", "틀림없이", "명백히",
    "분명", "확실",
    "절대", "결코", "무조건",
    "필연적", "불가피", "당연히", "당연", "자명", "필수", "필연",
    "확정적", "결정적", "단언컨대", "분명한", "명백한",
    "해야한다", "필요하다", "요구된다", "의무", "마땅히",
    "결국", "결론", "귀결", "요컨대",
    "중요", "핵심", "본질",
    "입증", "증명", "확인", "검증",
    "언제나", "항상", "모든", "전부",
    "매우", "극히", "현저히", "압도적",
]

ASSERTIVE_MULTI = [
    "명백하다", "확실하다", "틀림없다", "명확하다", "분명하다", "자명하다",
    "결론적으로", "결론은", "결론적으로 볼 때", "결론을 내리면",
    "종합적으로 판단하면", "결국은", "결국에는",
    "증명된다", "입증된다", "검증된다", "확인된다", "드러난다",
    "분석 결과", "연구 결과",
    "핵심은", "중요한 점은", "본질은", "관건은", "결정적인 것은",
    "분명한 사실은", "명백한 사실은", "틀림없는 사실은",
    "의심의 여지가 없다", "반박할 수 없다", "확정적으로 말하면",
    "반드시 필요하다", "필수적이다", "요구된다",
    "강조되어야 한다", "반드시 해야 한다", "실행해야 한다",
    "항상 그렇다", "언제나 그렇다", "예외는 없다", "반드시 그렇다",
    "따라서 알 수 있다", "이를 통해 알 수 있다",
    "이로부터 도출된다", "결론지을 수 있다",
    "시사하는 바는 분명하다", "명확하게 보여준다",
    "충분히 설명된다", "타당하다고 볼 수 있다", "강하게 지지된다",
    "반드시 추진해야 한다", "시급히 해결해야 한다",
    "더 이상 미룰 수 없다", "분명한 방향은", "우선적으로 고려해야 한다",
]

CERTAINTY_KEYWORDS = [
    "반드시", "확실히", "분명히", "명백히", "틀림없이",
    "확정적으로", "자명하게", "명료하게",
    "입증", "증명", "검증", "확인", "실증", "증거", "근거",
    "명확", "분명", "자명", "명백", "확실",
    "절대", "결코",
    "필연적", "불가피", "당연",
    "사실상", "객관적", "실질적",
    "타당", "유효", "신뢰성",
    "결정적", "결론적",
    "언제나", "항상", "예외없이",
]

CERTAINTY_MULTI = [
    "결정적이다", "명확하다", "확정적이다", "분명하다", "명백하다",
    "확실하다", "틀림없다", "자명하다",
    "증명된다", "입증된다", "검증된다", "확인된다", "드러난다",
    "실증적으로 확인된다",
    "연구 결과 확인된다", "분석 결과 확인된다",
    "자료가 보여준다", "데이터가 보여준다", "통계적으로 유의하다",
    "유의미하다", "유의미한",
    "객관적으로", "사실로 확인된다", "명백한 사실이다",
    "부정할 수 없다",
    "의심의 여지가 없다", "반박할 수 없다",
    "논란의 여지가 없다", "틀림없는 사실이다", "사실이다",
    "결론적으로", "확인된다", "알 수 있다",
    "드러난다", "증거가 된다", "증거이다",
    "충분히 설명된다", "강하게 지지된다",
    "타당하다고 판단된다", "설득력 있게 보여준다",
    "본질적으로", "근본적으로", "핵심적으로",
    "유의미한 결과를 보인다", "통계적으로 검증된다",
    "경험적으로 확인된다", "실증적으로 입증된다",
    "자료에 의해 뒷받침된다",
    "항상 성립한다", "예외 없이 적용된다",
    "보편적으로 나타난다", "일관되게 나타난다",
    "반드시 필요하다", "명백한 방향이다",
    "확실한 해결책이다", "필연적인 결과이다",
]

# ──────────────────────────────────────────────────────────────
# [v2 신규] 축4 단정형(J) 추가 지표:
# '~음', '~임', '~ㅁ'으로 끝나는 명사형 종결 어미 패턴
# 예: "문제임", "사실임", "한계임", "중요함", "필요함",
#     "해결됨", "증명됨", "확인됨"
# → 서술형 종결 문장을 명사화하여 단호하게 끊는 단정 문체의 전형
# ──────────────────────────────────────────────────────────────
# (아래 패턴은 calc_axis4 내부에서 직접 컴파일하여 사용)
NOM_ENDING_PAT = re.compile(
    r"(이?음|이?임|이?ㅁ"
    r"|함|됨|였음|있음|없음|해야함"
    r"|인듯|인셈|인것|인바)$"
)

AXIS_COLORS = {
    "axis1": "#FF6B6B",
    "axis2": "#4ECDC4",
    "axis3": "#FFD93D",
    "axis4": "#A78BFA",
}


# ============================================================
# 1. 전처리
# ============================================================

def preprocess(text: str) -> dict:
    text = text.replace("\n", " ")
    raw_sents = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in raw_sents if len(s.strip()) > 2]
    eojeols   = text.split()
    clean_ejs = [re.sub(r'[.,!?\"\'()【】]', "", w) for w in eojeols]
    clean_text = re.sub(r'[.,!?\"\'()【】]', " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return {"sentences": sentences, "eojeols": eojeols,
            "clean_ejs": clean_ejs, "clean_text": clean_text}


# ============================================================
# 2. 지표 유틸
# ============================================================

def _cs(clean_ejs, keywords):
    return sum(1 for w in clean_ejs if any(k in w for k in keywords))

def _cm(clean_text, expressions):
    return sum(clean_text.count(e) for e in expressions)

def _r(count, total):
    return count / total if total > 0 else 0.0

def _norm(a, b):
    t = a + b
    return (a/t, b/t) if t > 0 else (0.5, 0.5)

def _avg_len(sentences):
    ls = [len(s.split()) for s in sentences]
    return float(np.mean(ls)) if ls else 0.0

def _std_len(sentences):
    ls = [len(s.split()) for s in sentences]
    return float(np.std(ls)) if len(ls) > 1 else 0.0

def _concise_ratio(sentences, thr=6):
    # [v2] 기준 하향: 6 어절 이하만 '압축형 단문'으로 인정 (기존 8)
    short = sum(1 for s in sentences if len(s.split()) <= thr)
    return _r(short, len(sentences))


# ============================================================
# 3. 4개 축 계산
# ============================================================

def calc_axis1(clean_ejs, clean_text, total):
    """
    축 1: 감성형(E) vs 논리형(L)
    """
    ec  = _cs(clean_ejs, EMOTION_KEYWORDS)
    evc = _cs(clean_ejs, EVAL_ADVERBS)
    lc  = _cs(clean_ejs, LOGICAL_KEYWORDS)
    lmc = _cm(clean_text, LOGICAL_MULTI)

    adj_pat = re.compile(
        r"(스럽다|롭다|답다|스러운|로운|다운"
        r"|아름다운|아름답다|슬픈|슬프다|기쁜|기쁘다"
        r"|따뜻한|따뜻하다|차가운|차갑다|포근한|서늘한"
        r"|작은|작다|큰|크다|높은|높다|낮은|낮다"
        r"|긴|길다|짧은|짧다|넓은|넓다|좁은|좁다"
        r"|노란|노랗다|빨간|빨갛다|파란|파랗다|하얀|검은"
        r"|외로운|외롭다|불안한|우울한|행복한)$"
    )
    adj_count = sum(1 for w in clean_ejs if adj_pat.search(w))
    adj_ratio = _r(adj_count, total)

    CONJ_ONLY = {
        "따라서", "그러므로", "반면", "그러나", "한편", "다만",
        "결과적으로", "그래서", "또한", "더불어", "아울러",
        "게다가", "그럼에도", "그렇지만", "왜냐하면",
    }
    conj_count = sum(1 for w in clean_ejs if w in CONJ_ONLY)
    conj_ratio = _r(conj_count, total)

    adj_bonus  = max(0.0, adj_ratio  - conj_ratio) * 0.5
    conj_bonus = max(0.0, conj_ratio - adj_ratio)  * 0.5

    er  = _r(ec, total)
    evr = _r(evc, total)
    lr  = _r(lc, total)
    lmr = _r(lmc, total)

    es = er*0.35 + evr*0.25 + adj_ratio*0.25 + adj_bonus*0.15
    ls = lr*0.40 + lmr*0.30 + conj_ratio*0.20 + conj_bonus*0.10

    if es < 0.01 and ls < 0.01:
        _sents = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
        _avg = _avg_len(_sents) if _sents else 10
        es = 0.04 if _avg >= 10 else 0.0
        ls = 0.04 if _avg <  10 else 0.0

    en, ln = _norm(es, ls)

    return {
        "emotion_score": es, "logical_score": ls,
        "type":  "E" if es >= ls else "L",
        "label": "감성형" if es >= ls else "논리형",
        "radar": en,
        "evidence": {
            "emotion_words": [w for w in clean_ejs if any(k in w for k in EMOTION_KEYWORDS)][:8],
            "eval_adverbs":  [w for w in clean_ejs if any(k in w for k in EVAL_ADVERBS)][:5],
            "logical_words": [w for w in clean_ejs if any(k in w for k in LOGICAL_KEYWORDS)][:8],
            "adj_count":     adj_count,
            "conj_count":    conj_count,
            "adj_ratio":     round(adj_ratio, 3),
            "conj_ratio":    round(conj_ratio, 3),
        }
    }


def calc_axis2(sentences, clean_ejs, clean_text, total):
    """
    축 2: 서사형(S) vs 압축형(C)

    [v2 변경]
    - 서사형(S) 기준: 긴 문장 기준을 10 → 14 어절로 상향
      (14어절 초과 문장이 많을수록 S 방향으로 강하게 작동)
    - 압축형(C) 기준: 짧은 문장 기준을 8 → 6 어절로 하향
      (6어절 이하 단문이 많을수록 C 방향)
    - 가중치 조정: 문장 길이 지표의 S/C 기여 비중 상향
    """
    tc  = _cs(clean_ejs, TEMPORAL_KEYWORDS) + _cm(clean_text, TEMPORAL_MULTI)
    sc  = _cs(clean_ejs, SUMMARY_KEYWORDS)  + _cm(clean_text, SUMMARY_MULTI)
    lc  = _cs(clean_ejs, LOGICAL_KEYWORDS)

    tr  = _r(tc, total); sr = _r(sc, total); lr = _r(lc, total)
    al  = _avg_len(sentences); std = _std_len(sentences)

    # [v2] 압축형 짧은 문장 기준: 6 어절 이하
    cr  = _concise_ratio(sentences, thr=6)

    # ── TTR ─────────────────────────────────────────────────
    n_tokens = len(clean_ejs) or 1
    ttr = len(set(clean_ejs)) / n_tokens

    if n_tokens < 30:   ttr_weight = 0.0
    elif n_tokens < 60: ttr_weight = 0.4
    elif n_tokens < 100:ttr_weight = 0.7
    else:               ttr_weight = 1.0

    ttr_s_bonus = max(0.0, ttr - 0.85) * 2.0 * ttr_weight
    ttr_c_bonus = max(0.0, 0.70 - ttr) * 2.0 * ttr_weight

    # ── 서사 접속사 밀도 ─────────────────────────────────────
    NARRATIVE_CONJ = {
        "그리고", "그러다가", "그래서", "그러자", "그러면서",
        "그런데", "그렇게", "하지만", "그러나", "그후", "이후",
        "그때", "그러던", "그러고",
    }
    n_sents = len(sentences) or 1
    conj_count = sum(1 for w in clean_ejs if w in NARRATIVE_CONJ)
    conj_density = conj_count / n_sents
    conj_density_norm = min(conj_density, 1.0)

    # ── [v2] 문장 길이 지표 재조정 ───────────────────────────
    # 서사형: 14 어절 초과 문장 비율 (기존 평균 길이 보조 → 긴 문장 비율로 전환)
    long_sent_ratio = _r(
        sum(1 for s in sentences if len(s.split()) > 14), n_sents
    )
    # 표준편차 (긴 문장과 짧은 문장이 혼재 = 서사형 특징)
    sn_std  = min(std / 10.0, 1.0)

    # ── 최종 점수 ────────────────────────────────────────────
    # S: 시간 연결어(핵심) + 긴 문장 비율(상향) + 접속사 밀도 + 표준편차 + TTR 보너스
    ns = (tr                * 0.42   # 시간 연결어
          + long_sent_ratio * 0.25   # [v2] 긴 문장 비율 (신규 주요 지표)
          + conj_density_norm * 0.18  # 접속사 밀도
          + sn_std           * 0.10  # 문장 길이 표준편차
          + ttr_s_bonus)             # TTR 보너스

    # C: 요약 표현(핵심) + 짧은 문장 비율(상향) + TTR 낮음 + 접속사 부재
    cs = (sr               * 0.38   # 요약·압축 표현
          + cr              * 0.28   # [v2] 짧은 문장 비율 상향
          + ttr_c_bonus     * 0.20   # TTR 낮음 보너스
          + max(0.0, 0.3 - conj_density_norm) * 0.14)  # 접속사 부재

    sn2, cn = _norm(ns, cs)

    return {
        "narrative_score":    ns,
        "compression_score":  cs,
        "type":  "S" if ns >= cs else "C",
        "label": "서사형" if ns >= cs else "압축형",
        "radar": sn2,
        "evidence": {
            "avg_sent_len":    round(al, 1),
            "sent_len_std":    round(std, 2),
            "long_sent_ratio": round(long_sent_ratio, 3),
            "short_sent_ratio":round(cr, 3),
            "ttr":             round(ttr, 3),
            "conj_density":    round(conj_density, 3),
            "temporal_words":  [w for w in clean_ejs if any(k in w for k in TEMPORAL_KEYWORDS)][:6],
            "summary_words":   [w for w in clean_ejs if any(k in w for k in SUMMARY_KEYWORDS)][:6],
        }
    }


def calc_axis3(clean_ejs, clean_text, total, sentences=None):
    """
    축 3: 직관형(I) vs 구체형(D)
    """
    ABSTRACT_SUFFIXES = ("성", "화", "론", "적", "감", "도", "률", "력", "관", "주의")
    ABSTRACT_NOUN_SET = set(ABSTRACT_NOUNS)

    JOSA_PAT = re.compile(
        r"(이|가|을|를|은|는|의|에|에서|로|으로|와|과|도|만|라|이라|이며|이고"
        r"|에게|한테|께|부터|까지|보다|처럼|같이|만큼|마다|조차|나|이나)$"
    )

    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        tokens = kiwi.tokenize(clean_text)
        total_tok = len(tokens) or 1
        noun_tokens = [t for t in tokens if t.tag in ("NNG", "NNP")]
        verb_count  = sum(1 for t in tokens if t.tag == "VV")
        adj_count   = sum(1 for t in tokens if t.tag == "VA")
        noun_count  = len(noun_tokens)
        abstract_count  = sum(1 for t in noun_tokens
                              if t.form.endswith(ABSTRACT_SUFFIXES) or t.form in ABSTRACT_NOUN_SET)
        long_noun_count = sum(1 for t in noun_tokens if len(t.form) >= 4)
        sensory_count   = sum(1 for t in tokens if any(k in t.form for k in SENSORY_KEYWORDS))
        verb_ratio     = _r(verb_count,      total_tok)
        adj_ratio      = _r(adj_count,       total_tok)
        abstract_ratio = _r(abstract_count,  noun_count or 1)
        long_ratio     = _r(long_noun_count, noun_count or 1)
        sensory_ratio  = _r(sensory_count,   total_tok)
        abstract_words = [t.form for t in noun_tokens
                          if t.form.endswith(ABSTRACT_SUFFIXES) or t.form in ABSTRACT_NOUN_SET][:6]
        sensory_words  = [t.form for t in tokens if any(k in t.form for k in SENSORY_KEYWORDS)][:6]
        verb_words     = [t.form for t in tokens if t.tag == "VV"][:6]
        adj_words      = [t.form for t in tokens if t.tag == "VA"][:6]
        used_kiwi = True

    except ImportError:
        total_tok = total
        verb_ends = re.compile(
            r"(하다|되다|있다|없다|했다|한다|된다|겠다|습니다|합니다|됩니다"
            r"|았다|었다|는다|ㄴ다|았고|었고|하고|되고|있고|없고"
            r"|아서|어서|해서|되어서|느꼈|느껴|보였|이었|였다"
            r"|걸었|먹었|들었|만졌|움직였|시작했|두드렸|흩어졌|스쳤|들려왔|떠올랐)$"
        )
        adj_ends = re.compile(
            r"(스럽다|롭다|답다|없다|있다|같다|차갑다|따뜻하다|부드럽다"
            r"|스러운|로운|다운|없는|있는|같은|차가운|따뜻한|부드러운"
            r"|아름다운|아름답다|슬픈|슬프다|기쁜|기쁘다|외로운|외롭다"
            r"|작은|큰|높은|낮은|긴|짧은|노란|빨간|파란)$"
        )
        verb_count = sum(1 for w in clean_ejs if verb_ends.search(w))
        adj_count  = sum(1 for w in clean_ejs if adj_ends.search(w))
        noun_cands = [w for w in clean_ejs if not verb_ends.search(w) and not adj_ends.search(w)]
        noun_roots = [JOSA_PAT.sub("", w) for w in noun_cands]
        noun_roots = [r for r in noun_roots if len(r) >= 2 and re.match(r"^[가-힣]+$", r)]
        noun_count = len(noun_roots)
        abstract_count  = sum(1 for r in noun_roots
                              if r.endswith(ABSTRACT_SUFFIXES) or r in ABSTRACT_NOUN_SET)
        long_noun_count = sum(1 for r in noun_roots if len(r) >= 4)
        sensory_count   = _cs(clean_ejs, SENSORY_KEYWORDS) + _cm(clean_text, SENSORY_MULTI)
        verb_ratio     = _r(verb_count,      total_tok)
        adj_ratio      = _r(adj_count,       total_tok)
        abstract_ratio = _r(abstract_count,  noun_count or 1)
        long_ratio     = _r(long_noun_count, noun_count or 1)
        sensory_ratio  = _r(sensory_count,   total_tok)
        abstract_words = [r for r in noun_roots
                          if r.endswith(ABSTRACT_SUFFIXES) or r in ABSTRACT_NOUN_SET][:6]
        sensory_words  = [w for w in clean_ejs if any(k in w for k in SENSORY_KEYWORDS)][:6]
        verb_words     = [w for w in clean_ejs if verb_ends.search(w)][:6]
        adj_words      = [w for w in clean_ejs if adj_ends.search(w)][:6]
        used_kiwi = False

    _sents3 = sentences if sentences else [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    avg_sent_len3 = _avg_len(_sents3) if _sents3 else 10.0

    very_short_bonus = 0.08 if avg_sent_len3 <= 4  else 0.0
    very_long_bonus  = 0.08 if avg_sent_len3 > 12  else 0.0

    I_score = (abstract_ratio * 0.45
               + long_ratio   * 0.30
               + max(0.0, abstract_ratio - 0.35) * 0.15
               + very_short_bonus)

    D_score = (verb_ratio     * 0.30
               + adj_ratio    * 0.15
               + sensory_ratio * 0.40
               + very_long_bonus)

    ini, dn = _norm(I_score, D_score)

    return {
        "intuition_score": I_score, "concrete_score": D_score,
        "type":  "I" if I_score >= D_score else "D",
        "label": "직관형" if I_score >= D_score else "구체형",
        "radar": ini,
        "evidence": {
            "abstract_words":  abstract_words,
            "sensory_words":   sensory_words,
            "verb_words":      verb_words,
            "adj_words":       adj_words,
            "abstract_ratio":  round(abstract_ratio,  3),
            "long_ratio":      round(long_ratio,       3),
            "verb_ratio":      round(verb_ratio,       3),
            "adj_ratio":       round(adj_ratio,        3),
            "sensory_ratio":   round(sensory_ratio,    3),
            "avg_sent_len3":   round(avg_sent_len3,    1),
            "used_kiwi":       used_kiwi,
        }
    }


def calc_axis4(clean_ejs, clean_text, total, sentences=None):
    """
    축 4: 유보형(F) vs 단정형(J)

    [v2 변경]
    - '~음/~임/~ㅁ' 명사형 종결 어미 → 단정형(J) 추가 가산
      : "문제임", "필요함", "증명됨", "사실임" 같이
        서술 내용을 명사화하여 단호하게 마무리하는 패턴
      : 전통적 조사/어미 기반 hedge/assertive보다 더 직접적인
        단정 신호로 판단, J_score에 별도 비중 부여
    - 문장 길이 기준 재조정:
        긴 문장(F방향): 12 어절 초과 (기존과 동일)
        짧은 문장(J방향): 8 어절 이하 → 7 어절 이하로 상향
    """
    hsc = _cs(clean_ejs, HEDGE_KEYWORDS)
    hmc = _cm(clean_text, HEDGE_MULTI)
    asc = _cs(clean_ejs, ASSERTIVE_KEYWORDS)
    amc = _cm(clean_text, ASSERTIVE_MULTI)
    cc  = _cs(clean_ejs, CERTAINTY_KEYWORDS) + _cm(clean_text, CERTAINTY_MULTI)

    hr  = _r(hsc + hmc, total)
    ar  = _r(asc, total); amr = _r(amc, total); cr2 = _r(cc, total)

    sentences = sentences if sentences else [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    n_sents   = len(sentences) or 1
    avg_len   = float(sum(len(s.split()) for s in sentences) / n_sents)
    avg_len_norm = min(avg_len / 25.0, 1.0)

    # [v2] 짧은 문장 기준 7 어절 이하 (기존 8)
    short_ratio  = _r(sum(1 for s in sentences if len(s.split()) <= 7),  n_sents)
    long_ratio4  = _r(sum(1 for s in sentences if len(s.split()) > 12),  n_sents)

    # ── 조건/완화 종결 어미 (F) ──────────────────────────────
    cond_pat  = re.compile(r"(면|지만|는데|ㄴ데|지요|거든|잖아|걸요)$")
    cond_ratio = _r(sum(1 for w in clean_ejs if cond_pat.search(w)), total)

    # ── 선언 종결 어미 (J 기존) ──────────────────────────────
    decl_pat  = re.compile(r"[다요]$")
    decl_ratio = _r(sum(1 for w in clean_ejs if decl_pat.search(w)), total)

    # ── [v2 신규] 명사형 종결 어미 (J 강화) ─────────────────
    # '~음/~임/~ㅁ'으로 끝나는 어절을 탐지하여 단정형에 가산
    # 예: "문제임", "사실임", "필요함", "한계임", "해결됨"
    nom_end_count = sum(1 for w in clean_ejs if NOM_ENDING_PAT.search(w))
    nom_end_ratio = _r(nom_end_count, total)

    # ── 최종 점수 ────────────────────────────────────────────
    fs = (hr           * 0.38          # hedge 사전
          + avg_len_norm * 0.32        # 긴 문장 F
          + cond_ratio  * 0.20         # 조건 종결 어미
          + long_ratio4 * 0.10)        # 긴 문장 비율 보조

    js = (ar           * 0.22          # assertive 사전
          + cr2         * 0.13         # certainty 사전
          + amr         * 0.09         # assertive 다중어
          + short_ratio * 0.26         # 짧은 문장 J
          + decl_ratio  * 0.15         # 선언 종결 어미
          + nom_end_ratio * 0.15)      # [v2] 명사형 종결 어미 (신규)

    fn, jn = _norm(fs, js)

    return {
        "flexible_score": fs, "judging_score": js,
        "type":  "F" if fs >= js else "J",
        "label": "유보형" if fs >= js else "단정형",
        "radar": fn,
        "evidence": {
            "hedge_words":     ([w for w in clean_ejs if any(k in w for k in HEDGE_KEYWORDS)]
                                + [e for e in HEDGE_MULTI if e in clean_text])[:6],
            "assertive_words": ([w for w in clean_ejs if any(k in w for k in ASSERTIVE_KEYWORDS)]
                                + [e for e in ASSERTIVE_MULTI if e in clean_text])[:6],
            "nom_end_words":   [w for w in clean_ejs if NOM_ENDING_PAT.search(w)][:6],
            "avg_sent_len4":   round(avg_len, 1),
            "short_ratio":     round(short_ratio,  3),
            "long_ratio4":     round(long_ratio4,  3),
            "nom_end_ratio":   round(nom_end_ratio, 3),
        }
    }


# ============================================================
# 4. 최종 분류
# ============================================================

def classify(text: str) -> dict:
    d  = preprocess(text)
    s, e, c, ct = d["sentences"], d["eojeols"], d["clean_ejs"], d["clean_text"]
    total = len(c) or 1
    ax1 = calc_axis1(c, ct, total)
    ax2 = calc_axis2(s, c, ct, total)
    ax3 = calc_axis3(c, ct, total, sentences=s)
    ax4 = calc_axis4(c, ct, total, sentences=s)
    return {
        "type_code": ax1["type"] + ax2["type"] + ax3["type"] + ax4["type"],
        "axis1": ax1, "axis2": ax2, "axis3": ax3, "axis4": ax4,
        "total_eojeols": total, "total_sentences": len(s),
    }


# ============================================================
# 5. 페르소나 매핑
# ============================================================

PERSONA_MAP = {
    "ESIF": ("감성적 철학 에세이스트",
             "감정과 추상적 사유가 흐르듯 이어지며 독자를 깊은 내면으로 초대합니다.",
             "논리적 구조를 더하면 독자가 따라오기 더 쉬워집니다."),
    "ESIJ": ("열정적 인문 사색가",
             "감성과 직관이 넘치고 확신 있게 자신의 세계를 펼칩니다.",
             "근거를 보완하면 주장의 설득력이 높아집니다."),
    "ESDF": ("감성 서사가",
             "감각적 표현과 이야기 흐름으로 독자를 몰입시킵니다.",
             "때로 핵심을 먼저 제시하면 가독성이 올라갑니다."),
    "ESDJ": ("확신 있는 감성 스토리텔러",
             "풍부한 감각과 강한 주장이 결합된 개성 있는 문체입니다.",
             "유보 표현을 조금 더 활용하면 독자와 함께 생각하는 글이 됩니다."),
    "ECIF": ("짧고 깊은 감성 사상가",
             "간결한 문장 안에 추상적 감성이 응축되어 있습니다.",
             "구체적 사례나 장면을 더하면 독자의 이해가 쉬워집니다."),
    "ECIJ": ("단정한 감성 철학자",
             "짧고 단호하게 감성과 개념을 전달합니다.",
             "열린 표현을 더하면 독자의 공감 폭이 넓어집니다."),
    "ECDF": ("섬세한 감각 묘사가",
             "짧은 문장으로 감각과 감정을 정밀하게 포착합니다.",
             "좀 더 긴 호흡의 문장을 섞으면 리듬이 생깁니다."),
    "ECDJ": ("감성형 미니멀리스트",
             "간결하고 감각적이며 확신 있게 씁니다.",
             "추상적 사유를 더하면 글의 층위가 깊어집니다."),
    "LSIF": ("신중한 인문 탐구자",
             "논리와 직관을 서사 형식으로 조심스럽게 풀어냅니다.",
             "결론을 더 명확히 하면 독자가 논지를 파악하기 쉬워집니다."),
    "LSIJ": ("논리적 인문 사색가",
             "논리와 직관을 서사로 자신 있게 전개합니다.",
             "감성 표현을 더하면 독자와의 정서적 연결이 강해집니다."),
    "LSDF": ("신중한 현실 서사가",
             "사실과 감각을 논리적으로 흐름 있게 기록합니다.",
             "핵심 주장을 앞에 배치하면 가독성이 높아집니다."),
    "LSDJ": ("논리적 현실 분석가",
             "사실과 감각을 논리적·단정적으로 서술합니다.",
             "독자와 함께 생각하는 여지를 남기면 더 풍성한 글이 됩니다."),
    "LCIF": ("신중한 개념 정리가",
             "핵심 개념을 짧고 조심스럽게 정리합니다.",
             "구체적 예시나 이야기를 더하면 설득력이 높아집니다."),
    "LCIJ": ("T발C형 팩폭러 작가",
             "핵심을 짧고 단호하게 찌릅니다. 군더더기 없는 논리파 문체입니다.",
             "때로 감성과 유보 표현을 더하면 독자와의 간격이 좁아집니다."),
    "LCDF": ("냉철한 사회 관찰자",
             "사실과 감각을 논리적·간결하게 기록하며 결론을 조심스럽게 남깁니다.",
             "서사 흐름을 더하면 독자가 더 몰입하게 됩니다."),
    "LCDJ": ("정갈한 논리파 작가",
             "사실 중심, 간결, 단정적인 문체가 돋보입니다.",
             "감성과 직관을 조금 더 활용하면 글에 색이 더해집니다."),
}

def get_persona(code):
    return PERSONA_MAP.get(code, (
        "독창적 문체가",
        "16가지 유형 중 어느 하나로 단정짓기 어려운 복합적 문체입니다.",
        "자신만의 스타일을 계속 발전시켜 나가세요."
    ))


# ============================================================
# 6. 시각화 / 페르소나 상세 설명
# ============================================================

PERSONA_DETAIL = {
    "ESIF": {
        "emoji_title": "📚 ESIF — 감성적 철학 에세이스트 📚",
        "keywords": "#공감 #사유 #초대 #흐름",
        "intro": (
            "해당 유형은 독자의 손을 잡고 깊은 내면의 숲으로 걸어 들어가는 친절한 안내자입니다. "
            "지식을 주입하기보다 자신의 마음을 통과한 감정과 관념의 파편들을 흐르듯 연결하며, "
            "독자가 스스로 사색에 잠기도록 돕는 것에 깊은 보람을 느낍니다."
        ),
        "features_title": "📌 문체적 특징: 여운을 남기는 '동반형 문체'",
        "features": [
            ("자연스러운 전이와 흐름",
             "개인적인 일상이나 사소한 감상에서 이야기를 시작하여, 어느새 '존재', '정체성' 같은 거대하고 "
             "묵직한 철학적 질문으로 자연스럽게 연결하는 서사적 빌드업이 탁월합니다."),
            ("강요 없는 열린 문장",
             "결론을 성급하게 내리거나 정답을 쥐어주지 않습니다. \"우리는 어쩌면 각자의 섬에서 서로를 "
             "부르고 있는 것은 아닐까?\"와 같이 독자에게 질문을 던지며 사색의 공간을 넓혀줍니다."),
            ("한국 현대 에세이의 정수",
             "감성적인 문장 속에 묵직한 사유가 녹아 있어 대중적 공감대와 문학적 가치를 동시에 "
             "거머쥐는 가독성을 자랑합니다."),
        ],
        "limit_title": "😭 한계: 감성과 관념 사이에서 길을 잃을 위험",
        "limits": [
            ("약한 논리적 뼈대",
             "감정의 흐름과 추상적 어휘를 중심으로 글을 전개하다 보니, 글 전체를 관통하는 명확한 "
             "주장이나 논리적인 구조가 느슨해지기 쉽습니다."),
            ("모호한 메시지",
             "글을 다 읽고 나면 \"느낌은 참 좋은데, 그래서 필자가 하고 싶은 핵심이 뭐지?\"라며 "
             "고개를 갸웃거리는 독자가 생길 수 있습니다."),
        ],
        "tip_title": "💪 ESIF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'1문단 1메시지' 법칙",
             "각 문단을 마칠 때마다 \"이 문단에서 내가 하고 싶은 단 하나의 생각은 무엇인가?\"를 "
             "스스로 점검하고, 이를 문장 표면에 조금 더 명확하게 드러내 주어야 합니다."),
            ("구체성의 닻 내리기",
             "관념의 바다에 글이 표류하지 않도록, 추상적인 문장 뒤에는 반드시 눈에 보이고 손에 잡히는 "
             "구체적인 행동이나 사물의 이미지(예: '빛바랜 일기장', '식어버린 찻잔')를 연결해 "
             "글에 무게감을 더하세요."),
        ],
    },
    "ESIJ": {
        "emoji_title": "🔥 ESIJ — 열정적 인문 사색가 🔥",
        "keywords": "#확신 #열정 #철학 #선언",
        "intro": (
            "해당 유형은 감성과 사유로 가득 찬 내면의 세계를 뜨거운 확신으로 펼쳐 보이는 "
            "열정적인 인문학자입니다. 이야기의 흐름 속에서 자신이 도달한 철학적 결론을 "
            "거침없이 선언하며 독자를 압도합니다."
        ),
        "features_title": "📌 문체적 특징: 열정이 불꽃처럼 타오르는 '선언형 문체'",
        "features": [
            ("감성과 확신의 강렬한 결합",
             "감정적인 공감과 추상적 사유가 뒤섞이면서, 결론에서는 강렬하고 단호한 선언으로 "
             "귀결됩니다. 독자에게 거부하기 어려운 강한 인상을 남깁니다."),
            ("서사 속 직관적 도약",
             "이야기의 흐름 속에서 논리적 단계를 건너뛰어 직관적으로 도달한 깨달음을 확신에 "
             "찬 어조로 펼쳐냅니다. 읽는 이를 압도하는 지적 카리스마가 있습니다."),
            ("인문학적 깊이와 열정",
             "단순한 감상에 머물지 않고 인간 존재, 사회적 맥락, 철학적 개념으로 사유를 "
             "확장하며, 이를 뜨거운 열정으로 독자와 나눕니다."),
        ],
        "limit_title": "😭 한계: 근거보다 앞서가는 선언",
        "limits": [
            ("논증의 부족",
             "확신에 찬 선언이 앞서다 보니, 그 결론에 도달하기까지의 논리적 근거나 "
             "구체적 사례가 부족해 설득력이 약해질 수 있습니다."),
            ("독자와의 온도 차",
             "필자의 열정과 확신이 너무 강하면, 같은 결론에 아직 도달하지 못한 독자는 "
             "공감하지 못하고 소외감을 느낄 수 있습니다."),
        ],
        "tip_title": "💪 ESIJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'근거 먼저, 선언 나중' 순서 의식하기",
             "강렬한 결론을 내리기 전에, 독자가 같은 방향으로 사유의 여정을 걸어올 수 있도록 "
             "논리적 단계와 구체적 사례를 충분히 제시하세요. 결론의 힘은 더욱 강해집니다."),
            ("독자를 동반자로 초대하기",
             "\"당신도 어쩌면 이런 순간을 경험해 보지 않았나요?\"처럼 독자를 필자의 사유 과정 "
             "안으로 끌어들이는 질문형 문장을 중간중간 배치하면 공감의 폭이 넓어집니다."),
        ],
    },
    "ESDF": {
        "emoji_title": "🖼️ ESDF — 감성 서사가 🖼️",
        "keywords": "#몰입 #묘사 #감수성 #여운",
        "intro": (
            "해당 유형은 글이라는 도화지에 언어로 생생한 그림을 그리는 화가입니다. "
            "논리적 인과관계보다 눈앞에 펼쳐지는 듯한 정밀한 장면과 그 속에서 요동치는 "
            "인간의 감정을 포착할 때 가장 빛을 발합니다."
        ),
        "features_title": "📌 문체적 특징: 시공간을 이동시키는 '소설적 문체'",
        "features": [
            ("입체적인 감각 묘사",
             "시각, 청각, 촉각 등 풍부한 감각적 어휘를 동원하여 독자를 글 속의 시공간으로 "
             "순식간에 납치합니다."),
            ("서사 중심의 흡인력",
             "에세이를 쓰더라도 한 편의 단편 소설을 읽는 듯한 극적인 이야기 흐름을 보여주며, "
             "인물과 사건의 배치 능력이 뛰어납니다."),
            ("태도의 개방성",
             "서사의 결말 부분에서 교훈을 억지로 짜내지 않고, 열린 태도로 독자가 스스로 "
             "감정을 음미할 수 있도록 깊은 여운을 남깁니다."),
        ],
        "limit_title": "😭 한계: 풍경 속에 묻혀버린 메시지",
        "limits": [
            ("주객전도의 위험",
             "장면 묘사와 분위기 연출에 힘을 너무 많이 쏟다 보면, 글을 통해 전달하고자 했던 "
             "본질적인 메시지나 주제 의식이 서사 속에 파묻혀 희미해질 수 있습니다."),
            ("서사 과잉으로 인한 피로",
             "긴밀한 사유 없이 상황과 사건의 나열만 지속되면 글의 밀도가 낮아지고 "
             "산만하다는 인상을 줄 수 있습니다."),
        ],
        "tip_title": "💪 ESDF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'의미 부여' 문장 배치",
             "화려한 서사와 묘사가 끝나는 지점에, 의도적으로 \"이 경험은 나에게 ~라는 의미였다\" "
             "혹은 \"그 장면을 통해 나는 ~를 배웠다\"와 같은 분석적 문장을 한두 줄 삽입하세요."),
            ("사유의 브레이크 밟기",
             "이야기가 폭주하지 않도록 서사 중간중간 흐름을 잠시 멈추고, 현재 시점에서의 "
             "필자의 철학적 성찰을 섞어주는 완급 조절이 필요합니다."),
        ],
    },
    "ESDJ": {
        "emoji_title": "📜 ESDJ — 확신 있는 감성 스토리텔러 📜",
        "keywords": "#카리스마 #몰입 #결단 #교훈",
        "intro": (
            "해당 유형은 드라마틱한 이야기로 독자를 흠뻑 울린 뒤, 강력한 한 방의 메시지로 "
            "정신을 번쩍 들게 만드는 탁월한 연출가입니다. 생생한 경험담을 무기로 삼아 "
            "독자의 감정을 완벽히 장악하고 주도합니다."
        ),
        "features_title": "📌 문체적 특징: 가슴을 치고 뇌리에 박히는 '반전형 문체'",
        "features": [
            ("서사에서 선언으로의 급변",
             "감각적인 묘사와 생생한 이야기로 독자를 완전히 몰입시킨 뒤, 글의 종착지에서 "
             "거침없이 강렬한 결론을 내립니다."),
            ("확신에 찬 깨달음",
             "\"그 순간 나는 깨달았다. 이것이 바로 삶의 본질이다.\"처럼 자신의 경험에서 도출된 "
             "메시지를 절대적인 진리로 선포하는 문장력을 구사합니다."),
            ("폭발적인 감동과 설득력",
             "서사가 주는 정서적 감동과 주장이 주는 단호함이 결합되어, 독자에게 거부할 수 없는 "
             "강력한 카리스마를 전달합니다."),
        ],
        "limit_title": "😭 한계: 독자의 감정적 도약 실패",
        "limits": [
            ("설득 없는 강요",
             "필자가 겪은 주관적인 경험과 마지막의 거대한 선언 사이의 중간 징검다리가 부족하면, "
             "독자는 \"왜 저 경험이 저런 결론으로 튀지?\"라며 당황할 수 있습니다."),
            ("계몽주의적 태도",
             "선언의 어조가 너무 과하면 독자에게 교훈을 훈계하거나 강요하는 것처럼 느껴져 "
             "비판적인 독자의 반발을 사기 쉽습니다."),
        ],
        "tip_title": "💪 ESDJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'공감의 징검다리' 놓기",
             "\"이것이 바로 삶의 본질이다\"라고 외치기 전에, \"어쩌면 당신도 나와 같은 순간을 "
             "마주한 적이 있을 것이다\"와 같이 독자를 필자의 깨달음의 궤적 안으로 동참시키는 "
             "완충 문장을 배치하세요."),
            ("서사적 인과관계 강화",
             "자신이 내린 단호한 결론이 독자에게도 자연스럽게 도달할 수 있도록, 경험을 서술할 때 "
             "심리적 변화나 인과적 단계를 조금 더 촘촘하게 빌드업해야 합니다."),
        ],
    },
    "ECIF": {
        "emoji_title": "✒️ ECIF — 짧고 깊은 감성 사상가 ✒️",
        "keywords": "#압축 #시적 #여백 #아포리즘",
        "intro": (
            "해당 유형은 긴 말을 아끼고 단 한 문장의 칼날로 마음의 심장을 찌르는 시인입니다. "
            "구구절절한 설명보다는 고도로 정제되고 압축된 단어 속에 자신의 감정과 "
            "추상적 철학을 담아내는 능력이 탁월합니다."
        ),
        "features_title": "📌 문체적 특징: 문장 자체가 예술이 되는 '아포리즘 문체'",
        "features": [
            ("시적인 문장 압축력",
             "문장의 길이를 극도로 줄이고, 불필요한 수식어를 과감히 쳐내어 단어 자체의 "
             "무게감을 극대화합니다."),
            ("잠언적 파괴력",
             "\"외로움은 결국 자아의 윤곽을 선명하게 만든다.\"처럼 한 문장만 떼어내어 SNS에 "
             "공유하고 싶을 만큼 세련되고 깊이 있는 문장을 툭툭 던집니다."),
            ("조심스럽고 깊은 여백",
             "강요하지 않는 태도로 툭 던져진 문장 뒤에 거대한 사유의 여백을 남겨놓아, "
             "독자가 스스로 그 여백을 채워나가도록 유도합니다."),
        ],
        "limit_title": "😭 한계: 불친절함과 난해함의 경계",
        "limits": [
            ("맥락의 거세",
             "너무 문장을 압축하다 보니 친절한 맥락이나 부연 설명이 부족하여, 독자가 필자의 "
             "본뜻을 오해하거나 아예 이해하지 못하는 난해한 글이 될 수 있습니다."),
            ("추상성의 늪",
             "글 전체가 관념적이고 붕 떠 있는 단어들의 나열로만 이루어져 현실감 없는 "
             "'그들만의 리그'처럼 보일 위험이 있습니다."),
        ],
        "tip_title": "💪 ECIF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'압축 뒤 구체화' 패턴",
             "가슴을 울리는 멋진 시적 문장을 하나 던졌다면, 바로 다음 문장에서는 이를 일상적인 "
             "언어로 풀어낸 구체적인 장면이나 일화를 하나씩 덧붙여 주어야 합니다."),
            ("친절한 이정표 심기",
             "최소한 독자가 문맥을 따라올 수 있도록 문장과 문장 사이의 숨겨진 연결고리를 "
             "완전히 지우지 말고 조금은 남겨두는 미덕이 필요합니다."),
        ],
    },
    "ECIJ": {
        "emoji_title": "🗡️ ECIJ — 단정한 감성 철학자 🗡️",
        "keywords": "#선명 #단호 #잠언 #명확",
        "intro": (
            "해당 유형은 흐트러짐 없는 자세로 인생의 진리를 담백하게 일깨워주는 젊은 멘토입니다. "
            "복잡한 감정과 사유를 칼로 자르듯 명료하게 정리하여, 짧지만 단호한 문장으로 "
            "독자의 가슴에 깊은 인장을 새깁니다."
        ),
        "features_title": "📌 문체적 특징: 뼈 때리는 깨달음의 '잠언형 문체'",
        "features": [
            ("선명한 개념 전달",
             "\"행복은 결국 익숙함의 다른 이름이다.\"처럼 복잡한 감정적 현상을 한 줄의 명쾌한 "
             "정의로 시원하게 종결짓습니다."),
            ("짧고 타격감 있는 문장",
             "문장의 호흡이 짧아 가독성이 매우 높으며, 군더더기 없는 문체 속에서도 감성적인 "
             "온도가 유지되어 독자의 시선을 붙잡습니다."),
            ("철학적 주관의 명확성",
             "주저함 없는 문장력 덕분에 필자의 지적 주관이 매우 뚜렷하게 드러나며, 독자에게 "
             "강렬한 지적 자극과 신선한 충격을 선사합니다."),
        ],
        "limit_title": "😭 한계: 닫힌 대화와 반발 심리 유발",
        "limits": [
            ("독단적인 인상",
             "주관적 진리를 지나치게 단정적인 어조로 선언하면, 독자는 \"꼭 그렇지만은 않은데?\"라는 "
             "반발심을 갖게 되거나 필자가 독선적이라고 느낄 수 있습니다."),
            ("여백의 부재",
             "문장 자체로 상황을 종결시켜 버리기 때문에, 독자가 개입하여 함께 소통하고 대화할 수 있는 "
             "정서적 틈새가 닫혀버리기 쉽습니다."),
        ],
        "tip_title": "💪 ECIJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'초대형 완충어' 활용",
             "10개의 문장 중 1~2개 정도는 \"~이지 않을까\", \"~일 수도 있다\"와 같은 유보적이고 "
             "열린 표현을 의도적으로 섞어주세요. 주장의 날카로움은 살리면서도 독자를 따뜻하게 "
             "안아줄 수 있습니다."),
            ("다양성의 공간 열기",
             "자신의 정의가 '유일한 정답'이 아니라 '하나의 매력적인 시선'임을 글 속에서 은근히 "
             "내비치는 여유를 보여주어야 합니다."),
        ],
    },
    "ECDF": {
        "emoji_title": "📸 ECDF — 섬세한 감각 묘사가 📸",
        "keywords": "#정밀 #압축 #시산문 #이미지",
        "intro": (
            "해당 유형은 글을 통해 독자의 오감을 자극하는 천재적인 감각 수집가입니다. "
            "길고 장황한 설명 대신 정밀하게 엄선된 몇 가지 감각적 단어들의 조합만으로도 "
            "순식간에 글의 분위기와 무드를 완성해 냅니다."
        ),
        "features_title": "📌 문체적 특징: 오감을 깨우는 '스냅숏 문체'",
        "features": [
            ("감각적 이미지의 압축",
             "\"차가운 커피잔, 빗소리, 창밖의 노란 불빛.\"처럼 마치 카메라로 찍은 스냅숏 사진을 "
             "툭 던지듯 감각을 정밀하게 나열합니다."),
            ("시나 산문시에 가까운 리듬",
             "문장이 짧고 음악적인 리듬감이 있어 글을 읽는 것만으로도 특유의 정서적 무드에 "
             "흠뻑 젖어들게 만듭니다."),
            ("정서적 공명",
             "거창한 담론이 없어도 일상의 미세한 결을 묘사하는 것만으로 독자의 마음에 "
             "큰 파장을 일으킵니다."),
        ],
        "limit_title": "😭 한계: 현상에 머무르는 사유의 얕음",
        "limits": [
            ("사유로의 확장 실패",
             "멋진 감각적 이미지들이 화려하게 나열되기는 하지만, 그것이 단지 '분위기 잡기'에서 "
             "끝나버리고 더 깊은 내면의 철학이나 의미 있는 사유로 발전하지 못할 위험이 큽니다."),
            ("구슬은 서 말인데 보배가 되지 못함",
             "파편화된 이미지들만 남고 전체 글을 꿰뚫는 알맹이가 부족해 가벼운 감성 글로 "
             "치부되기 쉽습니다."),
        ],
        "tip_title": "💪 ECDF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'묘사+사유'의 2단계 문장 결합",
             "감각적인 묘사를 한 단락 늘어놓았다면, 그 밑에 반드시 \"그 불빛을 보며 나는 비로소 "
             "멈추는 법을 생각했다\"와 같이 그 감각이 이끌어낸 내면의 성찰과 사유를 최소 "
             "한 문장 이상 긴 호흡으로 묶어주세요."),
            ("호흡의 변주",
             "짧은 문장들 사이에 가끔은 생각을 길게 풀어내는 만연체 문장을 섞어 글의 "
             "리듬감을 다채롭게 만드세요."),
        ],
    },
    "ECDJ": {
        "emoji_title": "🏙️ ECDJ — 감성형 미니멀리스트 🏙️",
        "keywords": "#시크 #세련 #절제 #단호",
        "intro": (
            "해당 유형은 감정을 다루되 결코 척척하거나 질척이지 않는, 가장 현대적이고 "
            "스타일리시한 도시형 작가입니다. 감각과 감정의 핵심만 남긴 채 군더더기를 완전히 "
            "도려내고, 이를 아주 시크하고 확신 있게 마무리합니다."
        ),
        "features_title": "📌 문체적 특징: 군더더기 없는 '모던 시크 문체'",
        "features": [
            ("극도의 텍스트 다이어트",
             "감상적인 서사나 구구절절한 배경 설명은 과감하게 생략합니다. 핵심적인 감정과 감각만을 "
             "정제해 남겨둡니다."),
            ("단호하고 세련된 종결",
             "세련된 문장들 끝에 자로 잰 듯 단호한 어조로 마침표를 찍으며, 트렌디하면서도 "
             "이지적인 분위기를 풍깁니다."),
            ("감성과 확신의 결합",
             "감성적인 소재를 다루면서도 결론은 매우 뚝심 있고 선명하게 내리기 때문에, "
             "젊은 독자층에게 높은 선호도를 가집니다."),
        ],
        "limit_title": "😭 한계: 성급한 결론과 사유의 빈곤",
        "limits": [
            ("깊이의 부족",
             "감정과 감각을 포착하는 능력은 뛰어나지만, 그것이 왜 일어났고 우리 삶에 어떤 연속성을 "
             "가지는지에 대한 깊이 있는 탐구가 부족하여 글이 다소 가볍거나 성급해 보일 수 있습니다."),
            ("차가운 인상",
             "지나친 미니멀리즘과 단호함 때문에 글이 다소 건조하게 느껴지거나 인간미가 "
             "부족해 보일 위험이 있습니다."),
        ],
        "tip_title": "💪 ECDJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'왜?'라는 질문 던지기",
             "감각적인 결론을 내리기 전에 스스로에게 \"왜 이 감정이 나에게 이토록 중요하게 "
             "다가왔는가?\"를 한 번 더 묻고, 그 질문에 대한 내면의 사색 과정을 글 속에 "
             "반 단락이라도 심어주세요."),
            ("층위의 다양화",
             "겉으로 보이는 세련된 스타일 아래에, 인간의 보편적인 고뇌나 입체적인 사유의 레이어를 "
             "한 단계 더 깔아주면 글의 격이 한층 높아집니다."),
        ],
    },
    "LSIF": {
        "emoji_title": "🔬 LSIF — 신중한 인문 탐구자 🔬",
        "keywords": "#성실 #신중 #논리 #조심성",
        "intro": (
            "해당 유형은 서재에 앉아 밤새도록 문헌을 뒤적이며 진리를 탐구하는 학구적인 연구자입니다. "
            "철저히 논리적으로 사고하고 인문학적 개념을 집요하게 파고들지만, 자신의 결론이 "
            "오류일 가능성까지 염두에 두는 극도의 신중함을 보여줍니다."
        ),
        "features_title": "📌 문체적 특징: 학문적 신뢰감을 주는 '연구형 문체'",
        "features": [
            ("서사 속 개념 탐구",
             "탄탄한 서사의 흐름과 맥락 속에서 철학적 개념들을 차근차근 짚어나가는 "
             "성실함이 돋보입니다."),
            ("신중한 접근과 완곡한 표현",
             "\"이 현상은 어쩌면 우리 사회의 구조적 변화와 관련이 있을 수도 있다.\"처럼 조심스러운 "
             "어조를 사용하여 학술적인 신중함과 겸손함을 유지합니다."),
            ("탄탄한 지적 안정감",
             "글의 모든 구조가 논리적으로 잘 짜여 있어, 읽는 이에게 깊은 지적 신뢰감과 "
             "안정감을 선사합니다."),
        ],
        "limit_title": "😭 한계: 지나친 조심성이 만드는 우유부단함",
        "limits": [
            ("주장의 선명도 저하",
             "글 전체에 유보적인 표현이 너무 남발되면, 필자의 진짜 주장이 무엇인지 흐릿해져 "
             "설득력이 뚝 떨어집니다."),
            ("독자의 지루함",
             "너무 조심스럽게 돌다리를 두드리기만 하다가 글이 끝나버리면, 독자는 지적 카타르시스를 "
             "느끼지 못하고 지루함을 느낄 수 있습니다."),
        ],
        "tip_title": "💪 LSIF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'주장 선포, 유보 절제' 전략",
             "글의 핵심이 되는 주장은 칼로 베듯 명확하고 단호하게 선언하세요. 헷지(hedge) 표현은 "
             "부차적인 근거나 예외 상황을 설명할 때만 제한적으로 사용하는 균형 감각이 필요합니다."),
            ("마지막 문장의 임팩트",
             "결론 단락만큼은 \"일 수도 있다\" 대신 \"해야 한다\" 혹은 \"명백하다\"라는 단단한 "
             "마침표로 끝내는 연습을 하세요."),
        ],
    },
    "LSIJ": {
        "emoji_title": "⚙️ LSIJ — 논리적 인문 사색가 ⚙️",
        "keywords": "#이상적 #통찰 #완벽 #설득",
        "intro": (
            "해당 유형은 냉철한 머리와 뜨거운 서사 능력을 동시에 갖춘 비평가이자 칼럼니스트입니다. "
            "이야기의 흐름 속에서 논리적 근거를 차곡차곡 쌓아 올린 뒤, 마침내 도달한 완벽한 "
            "결론을 자신 있게 선포하는 지적 거장입니다."
        ),
        "features_title": "📌 문체적 특징: 깊이와 넓이를 모두 갖춘 '마스터피스 문체'",
        "features": [
            ("체계적인 논거 축적",
             "감정에 휘둘리지 않고, 서사의 전개 과정 속에서 주장을 뒷받침할 논리적 단서들을 "
             "빈틈없이 축적해 나갑니다."),
            ("자신감 넘치는 결론",
             "빌드업이 확실하기 때문에 결론을 내릴 때 주저함이 없으며, 매우 명확하고 당당한 "
             "어조로 핵심을 선언합니다."),
            ("학술 에세이와 비평의 이상향",
             "논리적인 설득력과 인문학적인 깊이, 그리고 읽는 재미까지 삼박자를 모두 갖추어 "
             "완성도 높은 평론의 표본이 됩니다."),
        ],
        "limit_title": "😭 한계: 차가운 이성이 만드는 정서적 바리케이드",
        "limits": [
            ("정서적 거리감",
             "글이 너무 완벽하고 논리적이다 보니, 독자가 감정적으로 비집고 들어갈 틈이 없어 "
             "글이 다소 차갑고 멀게 느껴질 수 있습니다."),
            ("공감의 결여",
             "이론과 논증은 완벽하지만 가슴을 울리는 따뜻함이 부족해, 독자의 '머리'는 설득해도 "
             "'마음'을 움직이기는 어려울 수 있습니다."),
        ],
        "tip_title": "💪 LSIJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'공감 선배치'의 기술",
             "딱딱한 논거와 데이터를 들이밀기 전에, 도입부에서 독자의 보편적인 일상 경험이나 "
             "감정에 호소하는 감성적 문장을 딱 하나만 먼저 배치해 보세요. 독자의 마음을 먼저 "
             "무장해제 시킬 수 있습니다."),
            ("스스로의 취약성 드러내기",
             "가끔은 완벽한 논리 뒤에 숨은 필자 자신의 인간적인 고민이나 서툰 감정을 슬쩍 "
             "노출해 주면 글에 엄청난 매력이 더해집니다."),
        ],
    },
    "LSDF": {
        "emoji_title": "🎞️ LSDF — 신중한 현실 서사가 🎞️",
        "keywords": "#기록 #팩트 #르포 #존중",
        "intro": (
            "해당 유형은 카메라를 들고 현장을 누비는 저널리스트이자 다큐멘터리 감독입니다. "
            "철저한 논리적 사고를 바탕으로 자신이 목격한 구체적인 장면과 객관적 사실들을 "
            "담담하게 기록하되, 판단은 오롯이 독자에게 맡겨둡니다."
        ),
        "features_title": "📌 문체적 특징: 사실의 힘을 보여주는 '르포르타주 문체'",
        "features": [
            ("사실 위주의 서사 기록",
             "주관적 감정 과잉을 철저히 배제하고, 눈앞의 사실과 구체적인 장면들을 논리적 흐름에 "
             "따라 묵묵히 기록합니다."),
            ("열린 결론과 독자 존중",
             "필자의 생각을 독자에게 강요하지 않고 비장한 태도로 결론을 열어두어, 독자 스스로가 "
             "현상을 보고 판단하게 만듭니다."),
            ("논픽션의 정석",
             "객관적인 팩트에 충실하면서도 서사성을 잃지 않아, 르포르타주 스타일의 에세이나 "
             "훌륭한 다큐멘터리 대본 같은 묵직함을 줍니다."),
        ],
        "limit_title": "😭 한계: 팩트 뒤에 숨어버린 주장",
        "limits": [
            ("전달력의 약화",
             "사실과 장면의 묘사가 너무 길어지면, 필자가 진짜 이 글을 통해 세상에 던지고 싶었던 "
             "핵심 주장이나 문제의식이 서사 속에 파묻혀 길을 잃게 됩니다."),
            ("주제 의식의 모호함",
             "독자 입장에서 \"상황이 심각한 건 알겠는데, 그래서 필자는 이에 대해 어떻게 생각하는 "
             "거지?\"라는 의구심을 남길 수 있습니다."),
        ],
        "tip_title": "💪 LSDF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'단락 종결 분석 코멘트' 기법",
             "객관적인 서사나 사실 나열 단락이 끝날 때마다, 그 단락의 맨 마지막 줄에 필자의 "
             "날카롭고 분석적인 코멘트를 한 문장씩 덧붙이세요. 예컨대 \"이 조용한 풍경이 방증하는 "
             "것은 결국 우리의 무관심이다.\" 같은 방식입니다."),
            ("핵심 주제의 전면 배치",
             "글의 초반부에 자신이 이 사실들을 왜 기록하기 시작했는지 문제의식을 명확히 "
             "밝혀두는 것이 좋습니다."),
        ],
    },
    "LSDJ": {
        "emoji_title": "📊 LSDJ — 논리적 현실 분석가 📊",
        "keywords": "#날카로움 #사회비평 #명백 #증거",
        "intro": (
            "해당 유형은 복잡하게 얽힌 사회 현상의 실타래를 단칼에 잘라내는 명쾌한 사회 평론가입니다. "
            "구체적인 현실의 증거와 팩트들을 논리적 자로 재단하여 누구도 반박할 수 없는 "
            "단호한 결론을 도출해 내는 데 귀재입니다."
        ),
        "features_title": "📌 문체적 특징: 진실을 규명하는 '송곳형 문체'",
        "features": [
            ("구체적 증거 기반의 분석",
             "오직 눈에 보이는 구체적인 데이터, 사실, 현장의 장면들을 논리적으로 연결해 나갑니다."),
            ("단호하고 명백한 결론",
             "\"이 장면이 보여주는 것은 명백히 계층 간 단절이다.\"처럼 증거를 바탕으로 아주 확신에 "
             "찬 어조로 결론을 내립니다."),
            ("비평문과 평론의 최강자",
             "논리적 빈틈이 없고 인과관계가 명확하여, 시사 평론, 사회 비평, 학술적 논평 글에서 "
             "압도적인 강점과 카리스마를 발휘합니다."),
        ],
        "limit_title": "😭 한계: 흑백논리와 해석의 독점 위험",
        "limits": [
            ("해석의 폐쇄성",
             "결론이 너무나 단호하기 때문에, 세상의 다양한 복잡성이나 다른 대안적 해석 가능성을 "
             "완전히 차단해 버리는 독선적인 글로 읽히기 쉽습니다."),
            ("독자의 반발 유발",
             "조금이라도 생각이 다른 독자들을 '틀린 것'으로 몰아세우는 듯한 인상을 주어 "
             "정서적인 반감을 사기 쉽습니다."),
        ],
        "tip_title": "💪 LSDJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'해석의 가능성' 인정하기",
             "\"이 현상에 대한 진단은 명백히 A다\"라고만 쓰지 말고, \"물론 이것을 B나 C의 시각으로 "
             "보는 관점도 존재한다. 그러나 본 필자가 주목하는 것은~\"과 같이 다른 해석의 존재를 "
             "짧게나마 언급해 주는 여유를 가지세요."),
            ("완곡한 수식어의 전략적 배치",
             "'명백히', '단연코' 같은 절대적인 부사를 조금 줄이는 것만으로도 훨씬 성숙한 글이 됩니다."),
        ],
    },
    "LCIF": {
        "emoji_title": "📑 LCIF — 신중한 개념 정리가 📑",
        "keywords": "#명료 #학술적 #요약 #조심성",
        "intro": (
            "해당 유형은 복잡하게 얽힌 사상가들의 이론을 깔끔하게 단권화해 주는 대학원생이나 "
            "든든한 학술적 조력자 같습니다. 거대하고 어려운 개념을 명료하고 간결하게 압축하여 "
            "정리하는 능력이 매우 뛰어납니다."
        ),
        "features_title": "📌 문체적 특징: 복잡함을 이기는 '네비게이터 문체'",
        "features": [
            ("논리적이고 간결한 개념 정리",
             "문장의 군더더기를 빼고 핵심 개념의 구조를 아주 명료하고 콤팩트하게 요약해 줍니다."),
            ("학술적 글쓰기의 모범",
             "사적인 감정이나 사설을 배제하고, 대상의 본질을 학술적이고 객관적인 태도로 다루기 때문에 "
             "교과서처럼 깔끔합니다."),
            ("신중하고 안전한 결론",
             "개념을 정리한 후 주장을 펼칠 때는 오류를 범하지 않기 위해 조심스럽고 객관적인 "
             "태도를 엄격히 유지합니다."),
        ],
        "limit_title": "😭 한계: 강의 노트에 머무르는 건조함",
        "limits": [
            ("추상적 관념의 나열",
             "개념 설명에 너무 치중한 나머지, 글 전체가 건조한 학술 용어나 추상적인 정의들의 "
             "나열로만 끝나버릴 수 있습니다."),
            ("필자의 목소리 부재",
             "남의 이론을 정리하는 능력은 탁월하지만, 정작 \"그래서 필자 당신의 생각은 무엇인가?\" "
             "에 대한 고유한 주장이 빠져 있어 개성 없는 요약본처럼 읽히기 쉽습니다."),
        ],
        "tip_title": "💪 LCIF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'개념+현실 사례'의 결합 법칙",
             "어려운 개념을 한 줄 정리했다면, 반드시 다음 문장에서는 독자들이 무릎을 탁 칠 만한 "
             "일상적이고 구체적인 비유나 사례를 하나 이상 강제로 연결하세요."),
            ("주체의식 드러내기",
             "글의 마지막 단락만큼은 다른 학자의 말을 빌리지 말고, 자신의 온전한 목소리로 "
             "\"따라서 우리는 ~해야 한다\"라고 명확하게 주장을 선언해 보세요."),
        ],
    },
    "LCIJ": {
        "emoji_title": "⚔️ LCIJ — T발C형 팩폭러 작가 ⚔️",
        "keywords": "#효율 #송곳 #핵심타격 #노필터",
        "intro": (
            "해당 유형은 글 쓰는데 1초의 시간도 낭비하지 않는 냉혹한 팩트 폭격기이자 언어의 "
            "외과의사입니다. 불필요한 미사여구나 감정적 위로는 완전히 도려내고, 오직 논리와 "
            "개념의 메스만으로 핵심을 가장 날카롭고 명쾌하게 찌릅니다."
        ),
        "features_title": "📌 문체적 특징: 감정을 거세한 '초효율적 팩폭 문체'",
        "features": [
            ("극단적 간결성과 개념 중심",
             "문장이 매우 짧고 단단하며, 오직 핵심적인 플롯과 논리로만 승부합니다."),
            ("자비 없는 단호함",
             "\"이 작품의 문제는 구조의 부재이다. 이상 끝.\"처럼 감정의 동요 없이 현상의 문제점을 "
             "정확하고 예리하게 해부해 버립니다."),
            ("압도적인 명쾌함",
             "지루하게 질질 끄는 부분이 전혀 없어 바쁜 현대인들에게 엄청난 카타르시스와 "
             "지적 명쾌함을 선사합니다."),
        ],
        "limit_title": "😭 한계: 권위적이고 차가운 지적 폭력의 위험",
        "limits": [
            ("정서적 거부감",
             "독자의 감정을 전혀 배려하지 않고 너무 날 것의 팩트만 들이밀기 때문에, 글이 지나치게 "
             "차갑고 권위적이라는 인상을 주기 쉽습니다."),
            ("소통의 단절",
             "반론을 원천 봉쇄하는 어조 때문에 독자는 글을 읽다가 마음의 문을 닫아버릴 수 있습니다."),
        ],
        "tip_title": "💪 LCIJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'쿠션 문장(샌드위치 화법)' 도입",
             "상대방을 칼로 찌르기 전에 푹신한 쿠션을 하나 까세요. \"물론 이러한 시도가 가치 있다는 "
             "의견도 존재한다. 하지만 구조적 측면에서 보면~\"처럼 독자나 상대방의 입장을 먼저 한 줄 "
             "인정해 준 뒤 날카로운 분석을 들이대면, 글에 품격과 여유가 더해져 설득력이 상승합니다."),
            ("비유적 부드러움",
             "날카로운 비판 사이에 위트 있는 유머나 부드러운 비유를 섞어주면 훨씬 매력적인 글이 됩니다."),
        ],
    },
    "LCDF": {
        "emoji_title": "🔍 LCDF — 냉철한 사회 관찰자 🔍",
        "keywords": "#저널리즘 #객관성 #현상포착 #담담",
        "intro": (
            "해당 유형은 차가운 유리에 비친 세상을 있는 그대로 기록하는 냉철한 관찰자입니다. "
            "주관적인 감정이나 선입견을 완벽히 차단하고, 사회적 현상과 구체적인 팩트들을 아주 "
            "간결하고 논리적인 문장으로 엮어내는 힘이 있습니다."
        ),
        "features_title": "📌 문체적 특징: 지적이고 정돈된 '저널리즘 문체'",
        "features": [
            ("간결하고 객관적인 기록",
             "화려한 수식어나 감정 과잉을 철저히 배제하고, 오직 검증된 사실과 구체적 팩트를 바탕으로 "
             "논리적인 서사를 구축합니다."),
            ("조심스럽고 객관적인 유보",
             "성급하게 한쪽 편을 들며 결론을 내리기보다, 현상을 다각도로 분석한 뒤 결론의 판단 여지를 "
             "조심스럽게 남겨둡니다."),
            ("고품격 저널리즘의 전형",
             "정갈한 기획 기사나 웰메이드 시사 다큐멘터리처럼 흐트러짐 없는 지적 세련미와 "
             "정확성을 자랑합니다."),
        ],
        "limit_title": "😭 한계: 동기 부여가 결여된 건조함",
        "limits": [
            ("정서적 몰입 실패",
             "분석은 정확하지만, 독자의 가슴을 뛰게 하거나 눈물을 흘리게 만드는 감정적 터치가 없다 "
             "보니 \"이게 나랑 무슨 상관이지?\"라며 글을 끝까지 읽어야 할 동기를 잃어버리기 쉽습니다."),
            ("기계적 중립의 함정",
             "너무 결론을 신중하게 남기려다 보면 글이 다소 방관자적인 태도로 보일 수 있습니다."),
        ],
        "tip_title": "💪 LCDF를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'감성적 훅(Hooking)' 배치",
             "딱딱한 본론으로 들어가기 전, 글의 도입부만큼은 독자가 감정적으로 확 몰입할 수 있는 "
             "구체적인 사람의 이야기, 혹은 가슴을 울리는 질문이나 생생한 장면 묘사로 시작해 보세요."),
            ("필자의 온기 한 스푼",
             "관찰한 현상이 우리 삶에 어떤 정서적 파장을 주는지 마지막에 슬쩍 언급해 주면 좋습니다."),
        ],
    },
    "LCDJ": {
        "emoji_title": "💎 LCDJ — 정갈한 논리파 작가 💎",
        "keywords": "#칼럼 #완벽 #신뢰 #팩트",
        "intro": (
            "해당 유형은 단 한 치의 오차도 허용하지 않는 엘리트 칼럼니스트이자 논설위원입니다. "
            "정확한 사실 정보와 구체적 수치, 철저한 논리 구조를 바탕으로, 짧고 단호한 문장들을 "
            "결합해 완벽한 신뢰성의 탑을 쌓아 올립니다."
        ),
        "features_title": "📌 문체적 특징: 신뢰성의 결정체인 '정통 칼럼 문체'",
        "features": [
            ("사면초가의 논리 구조",
             "사실 중심의 구체적 증거들을 간결하고 단정적인 문장으로 연결하기 때문에, "
             "누구도 흠잡을 수 없는 탄탄한 설득력을 가집니다."),
            ("군더더기 없는 정갈함",
             "감정적 낭비나 불필요한 미사여구가 전혀 없어 글이 매우 깨끗하고 가독성이 높습니다."),
            ("비즈니스와 언론의 이상향",
             "보고서, 기획서, 정통 칼럼, 논평 등 사회적 영향력을 발휘하고 신뢰감을 주어야 하는 "
             "모든 글쓰기 영역에서 가장 이상적인 문체로 손꼽힙니다."),
        ],
        "limit_title": "😭 한계: 인간미와 개성이 메마른 AI 같은 글",
        "limits": [
            ("기억에 남지 않는 개성",
             "글은 너무나 훌륭하고 정답인데, 필자만의 독특한 개성, 인간적인 향취, 혹은 예술적인 "
             "감성이 부족하여 독자의 뇌리에 강렬하게 기억되기 어렵습니다."),
            ("딱딱한 텍스트 덩어리",
             "정보 전달력은 최상이지만, 예술적 유희나 감성적 재미를 추구하는 독자에게는 "
             "다소 딱딱한 논문처럼 읽힐 수 있습니다."),
        ],
        "tip_title": "💪 LCDJ를 위한 글쓰기 벌크업 전략",
        "tips": [
            ("'감성 단락'의 의도적 삽입",
             "글의 논리적 구조를 해치지 않는 선에서, 본문 중간에 필자의 개인적인 에피소드나 "
             "당시 느꼈던 아주 주관적이고 감각적인 표현(예: \"그날의 바람은 유독 매서웠다\" 등)을 "
             "딱 한 단락씩만 보석처럼 박아 넣으세요. 글 전체에 환상적인 색채와 온기가 더해집니다."),
            ("스토리텔링 기법 가미",
             "논리적인 주장을 전개할 때, 딱딱한 설명조 대신 한 편의 우화나 짧은 스토리를 인용해 "
             "부드럽게 윤활유를 쳐주는 것이 좋습니다."),
        ],
    },
}


def _build_persona_description_html(code: str, result: dict = None, persona_img_path: str = None) -> str:
    d = PERSONA_DETAIL.get(code)
    if not d:
        return ""

    accent = "#1a1a2e"

    def item_list(items):
        out = "<ul style='margin:8px 0 0 0; padding-left:18px; list-style:disc;'>"
        for title, desc in items:
            out += (
                f"<li style='margin-bottom:8px;'>"
                f"<strong style='color:{accent};'>{title}</strong> — {desc}"
                f"</li>"
            )
        out += "</ul>"
        return out

    img_block = ""
    if persona_img_path:
        import base64
        try:
            with open(persona_img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            ext = persona_img_path.rsplit(".", 1)[-1].split(" ")[0].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            img_block = (
                f"<div style='text-align:center; margin:20px 0;'>"
                f"<img src='data:{mime};base64,{img_b64}' "
                f"style='max-width:400px; border-radius:12px; "
                f"box-shadow:0 4px 16px rgba(0,0,0,0.10);'>"
                f"</div>"
            )
        except Exception:
            img_block = ""

    evidence_block = ""
    if result:
        ev1 = result["axis1"]["evidence"]
        ev2 = result["axis2"]["evidence"]
        ev3 = result["axis3"]["evidence"]
        ev4 = result["axis4"]["evidence"]
        fmt = lambda lst: ", ".join(lst) if lst else "없음"
        kiwi_tag = "kiwi 품사 분석" if ev3.get("used_kiwi") else "어절 근사"

        def ev_row(label, value, color="#1a1a2e"):
            return (
                f"<tr>"
                f"<td style='padding:5px 12px 5px 0; color:#555577; font-size:13px; "
                f"white-space:nowrap; vertical-align:top;'>{label}</td>"
                f"<td style='padding:5px 0; color:{color}; font-size:13px;'>{value}</td>"
                f"</tr>"
            )

        evidence_block = (
            f"<div style='margin-top:24px; border-top:2px solid #e8e8f0; padding-top:20px;'>"
            f"<div style='font-weight:bold; font-size:15px; color:{accent}; "
            f"border-left:4px solid #4ECDC4; padding-left:10px; margin-bottom:12px;'>"
            f"📊 분석 근거 데이터</div>"
            f"<div style='display:flex; gap:24px; flex-wrap:wrap;'>"

            f"<div style='flex:1; min-width:260px;'>"
            f"<div style='font-size:13px; font-weight:bold; color:#333355; margin-bottom:8px;'>"
            f"근거 어휘 <span style='font-size:11px; color:#888; font-weight:normal;'>({kiwi_tag})</span></div>"
            f"<table style='border-collapse:collapse; width:100%;'>"
            + ev_row("감정어",       fmt(ev1['emotion_words']),    "#FF6B6B")
            + ev_row("평가부사",     fmt(ev1['eval_adverbs']),     "#FF6B6B")
            + ev_row("논리어",       fmt(ev1['logical_words']),    "#4a90d9")
            + ev_row("시간 표현",    fmt(ev2['temporal_words']),   "#4ECDC4")
            + ev_row("압축 표현",    fmt(ev2['summary_words']),    "#4ECDC4")
            + ev_row("추상 명사",    fmt(ev3['abstract_words']),   "#e6b800")
            + ev_row("감각어",       fmt(ev3['sensory_words']),    "#e6b800")
            + ev_row("동사",         fmt(ev3['verb_words']),       "#888")
            + ev_row("형용사",       fmt(ev3['adj_words']),        "#888")
            + ev_row("유보 표현",    fmt(ev4['hedge_words']),      "#A78BFA")
            + ev_row("단정 표현",    fmt(ev4['assertive_words']),  "#A78BFA")
            + ev_row("명사형 종결",  fmt(ev4['nom_end_words']),    "#6d45d4")
            + f"</table></div>"

            f"<div style='flex:1; min-width:220px;'>"
            f"<div style='font-size:13px; font-weight:bold; color:#333355; margin-bottom:8px;'>수치 지표</div>"
            f"<table style='border-collapse:collapse; width:100%;'>"
            + ev_row("평균 문장 어절 수",   f"{ev2['avg_sent_len']} 어절")
            + ev_row("문장 길이 표준편차",  str(ev2['sent_len_std']))
            + ev_row("긴 문장 비율(>14)",   f"{ev2.get('long_sent_ratio', '-'):.3f}" if isinstance(ev2.get('long_sent_ratio'), float) else "-")
            + ev_row("짧은 문장 비율(≤6)", f"{ev2.get('short_sent_ratio', '-'):.3f}" if isinstance(ev2.get('short_sent_ratio'), float) else "-")
            + ev_row("총 어절 수",           str(result['total_eojeols']))
            + ev_row("총 문장 수",           str(result['total_sentences']))
            + ev_row("추상접미 비율",        f"{ev3['abstract_ratio']:.2f}")
            + ev_row("긴 명사 비율",         f"{ev3['long_ratio']:.2f}")
            + ev_row("동사 비율",            f"{ev3['verb_ratio']:.2f}")
            + ev_row("형용사 비율",          f"{ev3['adj_ratio']:.2f}")
            + ev_row("감각어 비율",          f"{ev3['sensory_ratio']:.2f}")
            + ev_row("명사형종결 비율",      f"{ev4.get('nom_end_ratio', 0.0):.3f}")
            + f"</table></div>"

            f"</div></div>"
        )

    html = (
        f"<div style='"
        f"max-width:860px; margin:24px auto 8px auto; "
        f"font-family:\"Noto Sans KR\",\"Malgun Gothic\",sans-serif; "
        f"font-size:14px; line-height:1.9; color:#222; "
        f"border:2px solid #e8e8f0; border-radius:14px; "
        f"padding:28px 32px; background:#fafafa;'>"

        f"<div style='font-size:20px; font-weight:900; color:{accent}; "
        f"text-align:center; margin-bottom:6px;'>{d['emoji_title']}</div>"

        f"<div style='text-align:center; font-size:13px; color:#7c5cbf; "
        f"letter-spacing:2px; margin-bottom:16px;'>"
        f"🔑 핵심 키워드: {d['keywords']}</div>"

        f"<div style='background:#f0f0f8; border-radius:8px; padding:14px 18px; "
        f"margin-bottom:4px; color:#333; font-size:14px;'>{d['intro']}</div>"

        + img_block

        + f"<div style='font-weight:bold; font-size:15px; color:{accent}; "
        f"border-left:4px solid #4ECDC4; padding-left:10px; margin-bottom:8px;'>"
        f"{d['features_title']}</div>"
        + item_list(d['features'])

        + f"<div style='font-weight:bold; font-size:15px; color:{accent}; "
        f"border-left:4px solid #FF6B6B; padding-left:10px; margin:20px 0 8px 0;'>"
        f"{d['limit_title']}</div>"
        + item_list(d['limits'])

        + f"<div style='font-weight:bold; font-size:15px; color:{accent}; "
        f"border-left:4px solid #A78BFA; padding-left:10px; margin:20px 0 8px 0;'>"
        f"{d['tip_title']}</div>"
        + item_list(d['tips'])

        + evidence_block
        + "</div>"
    )
    return html


def _find_persona_image(code: str):
    import glob
    search_dirs = ["/content", "."]
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        try:
            for fname in os.listdir(d):
                nl = fname.lower(); cl = code.lower()
                if nl.startswith(cl) and any(ext in nl for ext in [".png", ".jpg", ".jpeg"]):
                    return os.path.join(d, fname)
        except Exception:
            continue
    for pattern in [f"{code}.png", f"{code}.jpg", f"/content/{code}.png"]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


BG_CHART = "#ffffff"
PANEL_BG = "#f4f4f8"
TXT_DARK = "#1a1a2e"
TXT_GRAY = "#555577"
TRACK_BG = "#e0e0ee"

AXIS_DESCRIPTIONS = {
    "E": (
        "감성형(E)", "#FF6B6B",
        "필자는 자신의 정서적 반응을 언어로 즉각 표출합니다. "
        '"정말", "너무", "매우" 같은 강조 부사가 자주 등장하고, '
        '"따뜻하다", "먹먹하다", "벅차다" 같은 감정 형용사가 서술의 중심을 차지합니다. '
        "독자에게 논리적 설득보다 정서적 공명을 우선시하며, 주관적 감상이 사실 서술을 압도하는 경향이 있습니다.",
    ),
    "L": (
        "논리형(L)", "#cc3333",
        "필자는 감정 표현보다 정보와 논거를 우선합니다. "
        '"따라서", "반면", "결론적으로" 같은 논리 연결어가 많고, '
        '"구조", "맥락", "원인", "근거" 같은 분석적 명사가 핵심어로 등장합니다. '
        "주장과 근거를 명확히 연결하는 습관이 있으며, 독자에게 이성적 납득을 추구합니다.",
    ),
    "S": (
        "서사형(S)", "#4ECDC4",
        "필자는 경험과 사건을 시간 순서에 따라 풀어냅니다. "
        '"그때", "갑자기", "문득", "그러다가", "마침내" 같은 시간 흐름 연결어가 빈번하고, '
        "문장이 길고 호흡이 깁니다. 장면을 묘사하고 감정의 변화 과정을 보여주는 데 집중합니다.",
    ),
    "C": (
        "압축형(C)", "#2a9d8f",
        "필자는 불필요한 설명을 제거하고 핵심만 전달합니다. "
        '"즉", "결론적으로", "핵심은", "요약하면" 같은 압축 표현이 자주 등장하고, '
        "문장이 짧고 단정적입니다. 정보 밀도가 높고 6어절 이하의 단문이 많습니다.",
    ),
    "I": (
        "직관형(I)", "#e6b800",
        "필자는 현상 뒤의 개념과 원리에 주목합니다. "
        '"존재", "본질", "정체성", "담론", "사유", "이념" 같은 추상명사가 글의 골격을 이룹니다. '
        '특히 "-성", "-화", "-론", "-적" 같은 추상 접미사가 붙은 명사가 많이 등장합니다.',
    ),
    "D": (
        "구체형(D)", "#b38600",
        "필자는 실제로 보고 듣고 만질 수 있는 것들로 글을 채웁니다. "
        '"빗소리", "냄새", "손", "창문", "목소리" 같은 감각어와 '
        '"걷다", "먹다", "듣다", "만지다" 같은 행동 동사가 많습니다.',
    ),
    "F": (
        "유보형(F)", "#A78BFA",
        "필자는 자신의 판단에 여지를 남깁니다. "
        '"아마", "어쩌면", "~것 같다", "~일 수 있다", "어느 정도", "~듯하다" 같은 hedge 표현이 많습니다. '
        "단정을 피하고 독자에게 해석의 여지를 열어두는 태도입니다.",
    ),
    "J": (
        "단정형(J)", "#6d45d4",
        "필자는 자신의 견해를 명확하게 선언합니다. "
        '"반드시", "명백하다", "확실하다", "결론적으로" 같은 assertive 표현이 많고, '
        '"문제임", "사실임", "필요함" 같이 명사형 종결로 내용을 단호하게 마무리하는 경향도 나타납니다.',
    ),
}

AXIS_PAIR_TITLES = {
    ("E", "L"): "📌 감성형(E) vs 논리형(L)",
    ("S", "C"): "📌 서사형(S) vs 압축형(C)",
    ("I", "D"): "📌 직관형(I) vs 구체형(D)",
    ("F", "J"): "📌 유보형(F) vs 단정형(J)",
}


def _build_type_description_html(code: str) -> str:
    letters  = [code[0], code[1], code[2], code[3]]
    opp_map  = {"E": "L", "L": "E", "S": "C", "C": "S",
                "I": "D", "D": "I", "F": "J", "J": "F"}
    pair_keys = [("E", "L"), ("S", "C"), ("I", "D"), ("F", "J")]

    html = """
<div style="
    max-width: 860px;
    margin: 32px auto 0 auto;
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 14px;
    line-height: 1.8;
    color: #1a1a2e;
">
<div style="
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 2px;
    margin-bottom: 24px;
    color: #1a1a2e;
">⭐ 유형 설명 ⭐</div>
"""

    for letter, pair_key in zip(letters, pair_keys):
        opp = opp_map[letter]
        title = AXIS_PAIR_TITLES[pair_key]
        my_label, my_color, my_desc = AXIS_DESCRIPTIONS[letter]
        opp_label, opp_color, opp_desc = AXIS_DESCRIPTIONS[opp]

        html += f"""
<div style="margin-bottom: 28px; border-left: 4px solid {my_color}; padding-left: 16px;">
  <div style="font-size: 15px; font-weight: bold; color: #333355;
              margin-bottom: 14px;">{title}</div>

  <div style="margin-bottom: 12px;">
    <span style="
      display: inline-block;
      background: {my_color};
      color: white;
      border-radius: 6px;
      padding: 2px 10px;
      font-size: 13px;
      font-weight: bold;
      margin-bottom: 6px;
    ">{my_label} ← 내 유형</span>
    <div style="color: #222; margin-top: 4px;">{my_desc}</div>
  </div>

  <div style="margin-top: 10px; padding-top: 10px;
              border-top: 1px dashed #ccccdd;">
    <span style="
      display: inline-block;
      background: #f0f0f0;
      color: {opp_color};
      border: 1px solid {opp_color};
      border-radius: 6px;
      padding: 2px 10px;
      font-size: 13px;
      font-weight: bold;
      margin-bottom: 6px;
    ">{opp_label}</span>
    <div style="color: #555; margin-top: 4px;">{opp_desc}</div>
  </div>
</div>
"""

    html += "</div>"
    return html


def draw_chart(result: dict, save_path: str = "mbti_chart.png"):
    try:
        from IPython.display import display, Image as IPImage, HTML
        IS_COLAB = True
    except Exception:
        IS_COLAB = False

    code = result["type_code"]
    name, strength, tip = get_persona(code)

    axes_meta = [
        ("axis1", "감성형(E)", "논리형(L)"),
        ("axis2", "서사형(S)", "압축형(C)"),
        ("axis3", "직관형(I)", "구체형(D)"),
        ("axis4", "유보형(F)", "단정형(J)"),
    ]
    ax_keys  = [m[0] for m in axes_meta]
    colors   = [AXIS_COLORS[k] for k in ax_keys]
    values   = [result[k]["radar"] for k in ax_keys]
    labels_r = [result[k]["label"] for k in ax_keys]

    fig = plt.figure(figsize=(15, 6), facecolor=BG_CHART)
    fig.suptitle(
        f"문체 MBTI  [{code}]   {name}",
        color=TXT_DARK, fontsize=15, fontweight="bold", y=1.02
    )

    ax_bar = fig.add_axes([0.03, 0.10, 0.46, 0.82], facecolor=PANEL_BG)

    for i, ((key, left_lbl, right_lbl), val, color, res_lbl) in enumerate(
            zip(axes_meta, values, colors, labels_r)):
        y = 3 - i
        ax_bar.barh(y, 1.0, left=0, height=0.48, color=TRACK_BG, zorder=1, linewidth=0)
        ax_bar.barh(y, val, left=0, height=0.48, color=color, alpha=0.88, zorder=2, linewidth=0)
        ax_bar.barh(y, 1.0 - val, left=val, height=0.48, color=color, alpha=0.18, zorder=2, linewidth=0)
        ax_bar.text(-0.03, y, left_lbl, va="center", ha="right", color=color, fontsize=11, fontweight="bold")
        ax_bar.text(1.03, y, right_lbl, va="center", ha="left", color=TXT_GRAY, fontsize=10)
        ax_bar.text(val / 2, y, f"{val:.0%}", va="center", ha="center", color="#ffffff", fontsize=12, fontweight="bold")
        ax_bar.text(0.50, y + 0.33, f"→  {res_lbl}", va="center", ha="center", color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_CHART, edgecolor=color, linewidth=1.1))

    ax_bar.set_xlim(-0.38, 1.40)
    ax_bar.set_ylim(-0.55, 4.0)
    ax_bar.axis("off")
    ax_bar.set_title("  4개 축 분석 결과", color=TXT_DARK, fontsize=12, loc="left", pad=10)

    N        = 4
    angles   = [n / N * 2 * np.pi for n in range(N)]
    angles_c = angles + [angles[0]]
    vals_c   = values + [values[0]]
    radar_labels = [f"{m[1]}\n/ {m[2]}" for m in axes_meta]

    ax_r = fig.add_axes([0.54, 0.05, 0.44, 0.90], projection="polar", facecolor=PANEL_BG)

    for rv in [0.25, 0.5, 0.75, 1.0]:
        ax_r.plot(angles_c, [rv] * 5, color="#ccccdd", linewidth=0.8, zorder=1)
    for ang in angles:
        ax_r.plot([ang, ang], [0, 1], color="#ccccdd", linewidth=0.8, zorder=1)

    ax_r.fill(angles_c, vals_c, alpha=0.18, color="#7c5cbf", zorder=2)
    ax_r.plot(angles_c, vals_c, color="#7c5cbf", linewidth=2.2, zorder=3)

    for ang, val, color in zip(angles, values, colors):
        ax_r.plot(ang, val, "o", color=color, markersize=12, zorder=5,
                  markeredgecolor="#ffffff", markeredgewidth=1.5)
        ax_r.plot(ang, val, "o", color=color, markersize=6, zorder=6)
        ax_r.text(ang, val + 0.14, f"{val:.0%}", ha="center", va="center",
                  color=color, fontsize=8, fontweight="bold")

    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(radar_labels, color=TXT_DARK, fontsize=9)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["25%", "50%", "75%", "100%"], color=TXT_GRAY, fontsize=7)
    ax_r.set_ylim(0, 1)
    ax_r.grid(False)
    ax_r.spines["polar"].set_visible(False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_CHART)
    plt.close(fig)

    if IS_COLAB:
        display(HTML(_build_persona_description_html(code, result, _find_persona_image(code))))
        display(HTML("<br>"))
        display(HTML("<div style='text-align:center;'>"))
        display(IPImage(save_path))
        display(HTML("</div>"))
        display(HTML(_build_type_description_html(code)))
        print(f"\n✅ 저장 완료: {save_path}")
    else:
        print(f"✅ 저장 완료: {save_path}")
        print("   이미지 파일을 직접 열어 확인하세요.")


# ============================================================
# 7. 텍스트 결과 출력
# ============================================================

def print_result(result: dict):
    """코랩 환경에서는 아무것도 출력하지 않습니다 (HTML로 모두 표시됨)."""
    pass


# ============================================================
# 8. 메인 실행
# ============================================================

if __name__ == "__main__":

    SEP = "=" * 62

    print(SEP)
    print("  문체 MBTI 분석기  v2")
    print(SEP)
    print("\n[STEP 1]  분석할 감상문을 아래 빈 칸에 붙여넣고")
    print("          입력이 끝나면 Enter를 두 번 눌러주세요.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "내 문bti 확인!":
            print("\n  STEP 2에서 입력해 주세요. 텍스트를 먼저 작성해 주세요.\n")
            continue
        lines.append(line)
        if len(lines) >= 2 and lines[-1].strip() == "" and lines[-2].strip() == "":
            lines = lines[:-2]
            break

    user_text = " ".join(lines).strip()

    if not user_text:
        print("텍스트가 입력되지 않았습니다. 다시 실행해 주세요.")
    else:
        print(f"\n{SEP}")
        print("[STEP 2]  텍스트 입력 완료!")
        print("          아래 빈 칸에  내 문bti 확인!  을 입력하고 Enter를 누르세요.")
        print(SEP + "\n")

        while True:
            try:
                confirm = input(">>> ").strip()
            except EOFError:
                confirm = "내 문bti 확인!"
            if confirm == "내 문bti 확인!":
                break
            print("  '내 문bti 확인!' 을 정확히 입력해 주세요.")

        result = classify(user_text)
        draw_chart(result, save_path="mbti_chart.png")
        print_result(result)
