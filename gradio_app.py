"""
Gradio Interface for Student Performance Prediction
Customized to match the exact Semantic Recommender prototype layout:
- Left-aligned header
- Horizontal controls with top-right action button (#e3e8f1 soft slate theme)
- Clean card-based 'Prediction Results' section mirroring the prototype card grid
- Dual-color distribution plot (Olive #848408 & Maroon #800020)
- Predicts ONLY when the user clicks 'Find predictions' (no premature initial predictions)
"""
from pathlib import Path
import sys
import math
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline
from pipeline import PreExamFeatureExtractor, SentinelAndBoundsSanitizer, build_pipeline

import gzip, base64, io

MODEL_PATH = ROOT / "artifacts" / "model_pipeline.joblib"
DATA_PATH = ROOT / "data" / "student_performance.csv"

EMBEDDED_CSV_DATA = """H4sIAAYsyGoC/4192w4syXHcu75lNKhrZuWjLcuw3wjoCxbUWiBMkcLu0pD/3p0RkTVz1gYsEZK4Z+v0dNclL5GRUf/9v7z+6be//fP//m9//dsvv77+02+//fyXf/7pL3/8+Q8///LHn//y20//8vPrD7/8/L/+9Ne//fqP//7Tv/7TH//6y8+v//Trr3/6l7/86/Pvf/2Hv/7rv/35599+/ufXP/3555//jY/5x3//7Zef/vi3X3750x//9ueffuEf/sOff/r11z/89Mtvf/rjn/7tp9/+9Ne/3Cf/55/++D///Nd/+fX1h7/++lv+zD/89S//40///PPzHq//+qe//PTn+9N/19vzP/213u111nu/vL+eP3rZ+7zG21/9PV/9Za8z3o1jx8ufP3vGzpf1V8Tzj/sZup5HnFd7+es8f8ChE3+29/Mv9/PI52+9+6s9/+TP/8//2PP3OHQ9v/j85fO21xyvbfiReMcr//J4/mOthu7XfD8v4c+j1/OI9nrGPn/l+evtNV87apzhrWI+v+Xndex5mf781Xxs/vbzFu0dHOrPnz9DTz7Hn0fjm/bzv4anPv/c386hB1Pl65mbM1+eLxiv/Kd8iZYztesF4pXjdg46+X3+fDvHYpqapqm35xknZ/15sD0Pej72eVZ/59R6fnrOy+TY/IKTb/7M68zFsnfHLOcHzPzPfg8OHfiCZ+FmftQz+/lRkRP8/JXxDLVnCjl0cmjPhdo5pblA9vxTf17k+b+v3TVVfWEaY+XbubbKMyK/Cl+R87rrbXMCn+fa86Mnv+J5ruHDApP1fPmzPHqwYQlXvsjzDjvfnssy8NjnwVFD88+fecr/Y8/0LmyWfOx8/vIzQzk/3K39YANZ5KYLzMPzy/Nt+ONcrudlO4cGXuDkDnlxj+fHP0/FsKlpHVwt7FQbtffyZzr23zOJfN7oWOrIE6RDlTM3MHY+Y5+5dn3R4HJbbuxcgPzthvXKP8bHL33RmJiUnKDn2/EW2FQTp+Q5a1zS8TwduzkHPEfkYCKf//K8fh477uax8U4nf+05S8dw7nvPbZen+tmmVtt0GE7DM/b5euz8XNs8hbm8z1/Ona/HOl7qGfqs9LPnuJ9yPU+tO3Y7B+eJGnn4niV9nt8x944lmnjdU0d/0BxETsB5HnZw+J4XfOUxy2V6ZpvfPxte4TVXPvE5Phiyn6fmE72s2eRrPf9s+Zq5m/IH0rRg6POmzwHiIj12KS3i0YKeIcvXcHjSSq1nJjh0wkZh4z+z/7zr85aGfYKDN+vneZqeOX5mLE0HLOTAvDbM6d13c3MCsfk8lwqHKLdff+V21DAeb+6k5/8f2J0hU5Ymt+zDdBhDh0UwWh3DpnEduLtN5sGBe05RGv5cJa8z32AgcT40S9yMjgM0cvnxKe3NTYBFKru7Gvby40v4UFr9iT2dQ0d+Lk/T4raIXIE1c7JyeXva9lys56m7dt8aOIyRxvNZpxgwo52HNGR2alEXN4bn0Te8BQ3qc5x0SO9eWYuLiqP5fJnRPKcfxNt6PpVnZW0c4GfrjHSSj6PAfsoFTAP5eEiuwTJYz5Ons+yYwaAE7P71EcvxTZbbI18zl/zdZMl7Oj2946FzhNOL3CUwyjBkdOReRn8F/rrjHeGkcjtPeOe0OSN/jU/dDS7GseGeRXCaiQYPlU6/55rS5u8OM4qxzxaLhTM/0spgoWYuFOd+D2yJ6JqltGppCWo2p07TnrTMI00kXxYHzzBXIw2a1ZsuWA2ckpPeBVM530f2PuAPOZRWw3Me05IeLFnu8wXTZ+m89VTDvFh+xTMXe+KoTJ3+PPm71bs6rT58+clABkZjld1Lv8GBB77l4GiufAMev434AkeqPMkO+uNIUxO58RYMFJ3ZMzHPAzSlRieT+xOPH++hYMvS3dOMWqclbznvh2dpy5qks++Ib/RAHiZ3nP8M9wxuefNI8zAd7VKbeHDkrDw2L0yDO1bhWekvs29cKc9zkb7xdO38jJQmvipdAsfmpDy7I2OYrdiMoVxPD2U1zBjG9mcA/YjjizZsJGJYTYBzAjos0/Nf6Mvygdwm57Of7WA/RG6T58cPJ2ApLsGOqtjIcqdFHsbnfUceUsfG83vwrzP1hkfgVR8riSVQAIQlQMjHr3LGEc9THRHUlEVf2lT5pzrSznhl5fw/B88r2hn43Jyru/s9A6Znc7rRPdPh5Q4b6clr+X3hEQdjsQwVxDYYnlmhnm9YGJzR5yBjy+ZR3gwP86fLQ3gG5pHbz/PYY0VzqOPlc69m1MKhjq1+8t0ee6VwN734wFP311B6Os9/mcakYRvhLyI2WGnO9AJ0UXvQSFrgqa7dB5tf338afi0C5y9XP/dBTpMhPng2SrkdbeD8q/n51rCn8Dq5t7j6eupgcJYHYCHl2jpUSwHK8694TM7EXocvn5jLiZU3TL9VSJxTjv2RJtm4Sw0zQJfnn116Nuc0t+/zSbD7CwYa8aa8Iw+KAkLPnG3D5xxs0v7Zz3qo18fvtHnPtB2E5GlWFpOCWv7DxPKxmDMNX54/voB2V4417aoTdLnIrx4D8OrxZhyUU3/K6EbDjDgCJ8dcvZnS1nH2inqiY/9uQ7hNJ74QHTN9XWlL+eMxePDSzj3RV/Dg9Sn/nHmGlZUILhMimQNnCv/3Xop481DImsfCybP1yQsQcL2ZLMFSVwoZG4fc84fTW9KeTQXTrzR9+ixjUpMBUS4Uwu9cOfjSlbafuzSUO+X5sQzZ6coRdmX6iIhTs3p49hE46ZgMxBIbAf/O7adv4jpZpiN7IUJBBMWICZ+kBJppEBYEWwoRSsbpeCRizsZxXEDPD7VKNDfhABizdUOZ3gbMZr75a8qhNRiZhaEZXnD39aZEIJ+1JkNZh9mmifzEUr0tTFXkZl+uhWow+yGoI1Y9duMLMpl8UrXn72DmHRnPRIC+a6TRoeY8pjln2KU9vb++yXFMcopfT97DfSQrkbtktBrIZYrcw4+FjjSlE4vPQ9Lrc3KNnsk/jPeMJoJRxObcKy3pnXHQgdM/dZ46ARFDmvss9eJYuic/CqBdgAjd+/My+djJocy8HXHMgvF584zR8FvuZ85Sn4jvPCc6XTmjuMBpwM5b3KS9L7zbccZGcHnGvYxk6xl8uEl7Z8D3vMYduxXAb6WPJk/aOx2yIVtHpslwu30gmVVfJR+VxyQDFC05XjZypvJl9QYHb8UMMvEXWeiGVUXM1+q7Asbe09492yTsRiiGvWcZkSCW6ENRK1J9q2ntiiV+eOroTCMywj9GJ8lIkvAZ4h4NvfCRtn4UNhAKOzM3lgHoQiUMsFAa7JtyD4wFJDM4dDFSSPt/MmNHwD9wtmmBj7xKH3QJML1lfeza9DS2HGZMORG/G7fL4UHBFuw5N9zZw/njeE8EdDR/TERxCGpdx2FYMIl0Abya2i3KIs+sydJhQpjgGSUvZd2MPYG18ZtmwyE4AE42IZdIvzausfZTYzteLxD0OK3KwtGu6KO8Sp9PwA/oKvFLusg+BA5NApjcLXPSLKX9ieBKNcxA0AB+YTN9LkwjkpQMZw2m3zVdCQ89y0q7PjfTSWaGjUHNBEC1aguUfZn8M0YrMgQH220i9LfrWfqUvyLoisgOU8CZgWm/M3voBXBm4QVcOXcrm6lotc8g4JszdADlIGdFFLAJkSmj7KsxeTW6thOcJ9g3HsTHYXBnL0E3CAKC+3Djbw9hw3cV1mDyiWXn3MJsNcVAwGU4sYtgGHCaDJeJ+RB2IORdCFVf9FhI7GIxWjUcr4Wtlf+G9m1xZyJVSd/ucGl9Kb4UpqAX4IsF0qrNtGNjYNAWl2sHSDHSKFguAE7MgTcy4v2ABXhmlxJYxMdpdjK4yynIFS0PvC4kGwj/iTUvZfOZqOTfx9DdiLhgnwSPCl9nKfGPWvzduU+WMoXcpBureudegGPfNIKRn3wScZZVWfgryLynNurmXkdWsxHZFjy6YVlOblTuk82kKvLQnQBGwBxtKZ9emtBNOPLku0UD9IayTOGd52IkfRMM8cGfR1LFfYdQiEdKP09/i2oPd1671ppmTVlq32kaEavtF88T7Tn+l+G/XpQJdADMlF8ZOP80VIbaAoYaz5nnYj8HZhMfzBeCnXySMn6Qdcw1Vim9Oj+kIMo8+F5Tb4OmdsqrHSXSQ/UA+CXue2NSknPxQopMO+kIawF3K0HulohchpAE2k0Hiadj5brrezglZzFKjnEDaiAbaSJrO5u96EzSLs60pqa8ewvHyXiHQ1lNQUSbrq8TQEwLmyiZ1ZfL5MHKtgzUHKBPI9xMk0fbIAcaAHgXv52hKEsc6/PjzqR3G+tRaUvTLOPJvQJ/zpLzr0fwhFaUPvFdh99UobdzPRD+ObIPv5tk44i6EKruE5/PWsNm3pPhb2cV7YUzoacuxp/IUgZhbFdoKxiv13cx8wWO9ngAoysdQImITGe8x6GGHPcg+2g0ZU1hRS+IgMfJifjB6GbwpxCxKV5sMns8Uc4TBRjbaVPy2+8cYHo0tYx4Q8His7HkH+aLyDCHHWHoeYJN5ZOR4NyAM4WPLv9wOrGsziorU56OkQEcdQty7IcLgyrvYfEOAc1RCeXAenHoJMiBQKHxrKRBCRm0fTPKfhai+oNdh+Oy4SDsxVMq2KEfgqAHx3jmBtz69VVBfa1pvhy2qVwEQ+qu6pHOvp5Ku2fOQD2rYgw5jDXD2qeHASF2VPCbsPHiojOFJnThQevokTvrPAzykSRVOhNKqBDSL2bdTaeE1WCv/RzEt3J/A6Ag1s6wy5h0lnkOYXtpzPZg/SBoyRDK2cWFe9AHx2ZGl/AwElDgOI515jiGNSeP2uY3Idoxocg7fSO3SAjxg6l4XvQo6T4v5LurXpIBO4OignsGjlEFe9czB4FylEMMcNPGnq/6yrmfk1nhZrSblhSgwFLRzC/Q2xGIo76JJFpL2WkdFirbS8nGaCynnAEchQayK9Br9A0qso3GGgcee3YBqCagF8GGZmk0oiWGIoAh1sJmMrkH5KeLQ4kheYYWGT4x5SNpodHszhq6WOLCbNkHvz6KCeGKD4duLB9qsU68Pz8LlUwFsC6wcTQrU7aB37sSdNZScwVuajCETzjWZ9KaMw9bZUxnzQHnJiYCNIb7RhPlZSG/HqzMt8GqM5DyyjsJz1aNfWSwxBytAZ41edOv+qki3tF72Z6ZC3HoYAdK7BsufavcOfqA8UciwbpEoQrMY0z7dXRaGliKhX3AocEy7guVXf38QtoYqM61l8zW1EkQH0RP3TT8E1lyPhXlwDchPmG5Gsp9DKeeqb9zrsY92ogNuGe6s3jemHRFB6TECKQr8KsN3k/NAAuz7ZqBrYhqqzY4erBgm+/4HMU83/FmHsL6/bhZzwB3otMMGuAv4t5DMR2YCTyNg9PuAOrPS7jnwGIhP/N65oDFB06x07OKDRNK00Fr4VcNHjDUG7kGBkSr/D/8Pb9q8NQjndgoNyPbe7uqTvsClWMwqggEwAPVHM0p61jzEqLGMNhHxrSdAcjQyoqSY/VUf/19eqGdD3tcVVJiUG6lt03LNYNudQipOBPjeWKOcuQcupOYwG04mFEBJ54DpQIl8xXYbvmsMRnvIENbuwoqQG3Slr0uH2dMFswNZdbGCpXBJE6VHrPIzKGj9nbPVFJx1VIcOEjN4FfNiR80kr3IX6JvJJ6wbjI55vr4DTfmNEzTSElyxNwcuhGbnyGglpazM/t2vMFSRWdMJr5n0hKFqaywBID3dsleYzqWHIX3TEZdlI8prpmteqjoLhOOsvFkNXpNwEK0bDNYC2+ke0RZ+H2RusLexmL1wwAotPvMI1wVxQdhJKMgiiNkd95PSpsxyWHRYwd96mbyaUwU+w1EPudqzSLGLMRqJrdxLlJvwhLG4ko5oyYWX99M37BRt9VAAhRk0BijahJuQDdwpWljWfkWR1jlKs+a+HgnPSGXaN2qLyAHm68lFINJKvhYPKiLRV/E9Is1moFzIyJFphX7fn4UkQDlYRVpWH3i9Fc9bWw+woA23ZPSp2pf2KnlXLYSYHEDM8A674I2mbEVWDs2LejprGgiCd3pB458BnIpftueNELBE6Aj2AsyfSLDvWsoK40LviDTjvdSxqiCevns5KzB/pN2cboqJQwbCStzsjbDwYB1BSWQgfUmM+3UOd0sypAUGZNu+ABWPwLTPiFDOogIRhfA3zOlGvLXmb3UuID5xoE2IzepyGEKhcpQGlldsH75dQbcETk5DsqzPcUOGfZVqUq3QiRHzKwvn2Kj8Emk34gaWeaglYKn0q9PHB3QAxzVdFSzGuKfC+cPI5QOfCZUpeBKOEZ6zjOX0vitsE8GdgDocQDRil06RKBwOB7n4iBNZL21AJdhXgCWg0CwPymviy8Zuz7lMIydQtyJfZFgReqaGG7DAqgmORGkReCb7QN01vwAoQCZhoA3Ay6dC79A88j6arLGQP/sRUQcwk+zTqrEZwA3EM/lFaymh2qt2EN1fHBmgZ4NENGO9iVdO4oNZcMATQQC/Aw5XUkUdrQQ6SmkcTgpLqeJPeCX38Ziy7hF8QH+BNwSwLYEnarUgRLu89RT8ySOX07pbnwqcfohhtlWBXk4S9AbiOVmYXSIaMSsynq9a5RlIreZkQ6COkCdn2L3UIWJDGDAsltujGv6mddDkPoYk9nNnGMr60YQM2uoaC6L+NQJ0QaLWGs3mx5n4vUBtIig3ABihjwYDxKqIECjN6Bjng5TfPwDyWkcJlSJ/6RTtKUDFyQlZQgx69dZKzkAcae83VR0lmfqy4Ydrg1X9TA2pmM8H+YaV/WcKqFXuccuLIyn3sDosBC1nJzdQ5iJiA+mv8iNI4h+ZwE0d9Txy/O68GAFhtGLibpkJYhKH2WJiXdVvBd0TIC6M4KYXCKgPvtiY9zXQQgaHMd589/+DiXg/eOYcEA3YUdrTD5J6jlgxmDHcQqCnMYDLAGZD8gbKnuDPijW7ggxnAFzoTIJLMnfSkHZMsCNHSzvYWNnNjVEuAllnt+xITw33BBgr0IhhE8j6C73GCydH/iIzQWLdOT7QyE8Wt0pxMLRJgF3qtmyipEax3XmQ6gej0oljnZB/yKEz0bU8ASBKqHunTU3LxdtHMv1CtIY2eIwtQ86hwqvmW3xsxDLE9FGJV4h1XfqNZuonfG1ZfaN+xUiDQ4lE/IgAV1V8Zuij8NsT56Z2bzAFdT+d9d6cThy2l7fRVA9k2EcGr9k/KmTWLtgNmK0Zqw8eEUJDdiKaHz8rN6KTQPkOd01Es+m/oY8X3xoZ0NHNPYsnGpEsAuYVcFvdmaPnICB5pEipYJzBmLq5FAmjkCzu76IXnTxZGkYTxbK00PBbM+VEpE8x8plTAEV8G+wbQ7Yl9DeKZRcn89iR3pwHO4j2gMrr3CFCpJnFwqIdVzkb4/Lu/sBhZr9VJIIghJPS1OZqrPqpQ8LmJ1DsCDPS2inskg08gX4XSBUBPiC8MUM6IaC+vmFfsyhWmJjdGOiO5aFzUjRFQnNQXcKJI5tPsBo3kslSsKuHMr6xwau3BkKFRSYsOUHrJqD/odlAq5DGbh9cZ3aWoPlaAMKg4TpJQjuw46dg9Tmc5gnZa8H7HUgQ25M5zmpg2wfdkP1AtWAP5HCvm5NZw7mIoESzKkkvQswU42Q21ooBXCSpfiSHTJ8pMK7Ocl23v75dq98yqvJSxH9nB2JgeXavF5qF1pF+dMYbUfMu1VA3QRDi5jAd5ysBRfvCUTCQHFnFy9ZieecpKzBoOxFxktoQuEwM2TUUzcMwYRXtbvvXKX2SiRntXksskhUReW8XyoTd/MkAztzNTRkmEpuhft9qDlzsnQB1lXWHpaYiVPlbrt53JxMpTYwJzK/RPvyYrwIrJ6rVcMe2gDJAGTrUpnpAtPmYnKJ8zSAepnKGVt0j11Hf/FQBBttUB9/k0vEzD9TEL0ATV9gqwaje0NsvasXT2nnXET9z2Fmo2ra+JqBj+1Z6sjpIAlyB4zkUBRcfr4Hi/Q8BX2a6Ez+ST1VUJyLHKUYxH/96KBMhLl6XW6XdYrxvkBOJ6bbp5iPKilrbMDTHKNfxd5e8q18W1eAOXdjcBMIRSpy5qINwOqhvGXuXmUAbGx4IFO5rhejkXO7xXlpyjBubBOqwX1ofXMzMUZAPlXW27Ts5PXZZ9fuRaYkqjGDQCFTSVr2c8PRuZn85ArAAbMdb1SdsjbMNla1QdTamNkE6LZqCqd29vbPDvCojrDe3q6z9SGpzc0Q+dCw0bE2somX0v/Pa9LURuj3tbPGe1+LUfSgaSS6or7m/SUkZqrtAa9Qy2oqLS4yr5VjAmYR/7IAsGmDtKvN8FYtN0TVByer11NnMRXJZuULqP7txT/kKUi+d74sXKCo/0tbW0ih2JpTPR+OIsWsJpUhVrkKtnoDdk4EIoFg3H4QLBDXj1vYmCaHxbiF+6q/mpp+JkNhTQHbxVAv+Tz1mylcxI5ppD8vZ2yXFk6NVOxiq2LF9FY1fVPTz1H1nUc7bpY1nRFqoPrbyT4nCeOosO3i1U4QK05ulwGDMbWz/auJUfDzdML3gbYzPTZQgv99eXk+697A+1Ob0FbaQhIgMqdy2L6r6QsJMXfAm7CQCnZcKrcCJJBj4h9Jp6Yvnp+A1Zl+GnIPI1mBkfUgxiACwnQiTCDzTdAV5It2wRYaF6Q5braRhatia/KZ+/bczKOWmK4yxVDxJS71vcokEyy/JY7mIE12KXfv1UZHE3BG5YxbpIr8bzrS6MzTr0/WU0Bgz0N5MyDyfx7DVlblEKFFqeiZB9EKmBUkCwSUfI6k5QSlKXMqVg7PV1di9eXNY4hjiD5OLvzWfuo3addY9mij68WsSqDrMl8/ndPzqMcIoPoit4Dk8SOiikW9rDJ0Y0Xh7MtVJ6f6K2KKVm2MQN/Ph/h2BFveDR1q/2+Mr1DeLq6I07JXYpVsG4ciwXyJgNnVuj7Z98H9DMzCWauNXYzu9iYwq/4g/Tq5tIY+XDm0pZhpkqOs3+ZhqlpOsPo9tE/mJ/oO7ly4lHNRK+QY2vvF65ho+yBdOIliFtopZLJ2ZoDcUsHAkOnnJPtrqy+PUWj1ec+gn4LpX1FiCB0WlaSapT29RPtAUpd9J0vLNFVR9huor8Zy4kIbnb9UIDOW06gDMDiQpV/Gy4NgScd8RdUddfZXY2cfkThTllSN3pKtUOFxNaK+2VaIKEmJ1CXrjNtJudquMBxz5fMKHIyik5/6KgaArvzzjEujnqpTl+tbjU4qOjMlxVSIHgosuXN1WKLsTOicXJh2vT+esjmUAGUhzF6k66EeHXItMVT9H72R5VJ9Avp+Et+DI8n/q3YqJzU7z/P1O5ypPoobPXKrYMhUSIGw/tTAVF/IiJ298F157yCuyq/u/LzMJtZUAZ8uGLX5VS+3qyZbvV7A85rYPzJO+mSyK1DBck67qC9fBX/9urMYeIK1FHnqfk0+i9Ycexj2Y4uykQLZzGCnY7uh/OpqLAXz6ktYpAv427dWsQY3L971ye44T0u1uTxKhZOuwU5f9jj7FXYpanRjWyaPyOCOPGx6AJXhTcwqFMvebGap3QOkiyLwu+iJP/Y7rEHehmOjZacy8cmskHXSDWggBpN8EP0TyGNawvZJoW5R308sCZHZdraljtttmc2SSg3W4Esh9STZckqGwn+INtdgNzaBcmNd3AVRreqg069HIeVRc3okQ3DxudqBk0xaVjGzMukvUlARa4GTzu0nHYrdxSLmummbIoTgLM1R/bg9W112v+zAkj6pTrc1tUbB0GAPWbwCiGEd9eu0xgjLk0Cy1bgw1ePQb6y95mb1GNjHKJ9T8Rbck/LYNUV3RsxaqkImWR1yg4sZsyYnm5T4oUlt6vFXzBEqgK0p0A8FqsP9v8SkTu+8hf2vKZPXmRmepszQ1JDgF8xei4ELah9dT1zKYeOiuGsp2utUNjmER9FyLxBbjyO2bxB0MHbZFIdr/0DNXIskANbuG3mUXUwP/6HHZq1FPn5aqhddzRHprrERXr/OShDzUid9K75axnEcOfHr9k7Rg3WdkXMBhFOHaTGfQPfaQapfu+RGsKumiYtxsDeSAPBW1yfOZ1ySz1pcoeyjRcmlQjLScge7JjhPm4h1AMAE3xRD2WURDN/K2W46jrQYIlqxb7wr0vvolazNDkQgyBv4HFZD4aZYl3rqZFCL0zwPm0waAmPWfqFSRYNSz8klYH26qew5KtLkVG2KGViI51XkrQ/Rx5WRri0+EmBMJyygxEusT6/QaHvxbHpVHlHCEs2pfW/9ra5EkOE6hTC2NI12cbI0A1Hrys55siHOm98nd8KhxokhitwobUNXwHpyRpY6z8ZWMXKdNuvJLT1a/9j+MhSZuJUUhOuksEqL6RfQvUSgSLv5ehH3VTF/MELQMFYDUczfJgB5JxlGyBA7PXj6jKYvgIMtHpUuNQQrQho3ldn/H2dfxm2D6IqyL37p6P024Sy71V5D4YIpa7Bhxj+myaJ6lSBSouaWeWWHiBFhqHM3JCyTru4QDr7t1SRscYb8ro2rEZaV8a/OnrL3TiEsSJ+wbuVqqvMqon8iCJX1wLE1dQyKlsB9H58Nmp3/cqFNI6fcyKpWWL0CA8pF4iKhU/b0VV9fJdnLGadDROyoygE87lx76qLZLvfiJixQpuIl01d0Kc0VI4CN4ppRBIAhnKsat5RfLnXTns3+vwrMgI+qGPSZq0PaLKosmWMvPG/dlBChIN/0XPwcfX0RNyUypbjX6R/Cz6DDbXQYIw1/h/BzWGl+2FGWa5QnO1t03CnJiARdNK1o8dCJMn/dsulWpfnmGEfSBlUOUk/f26/in9cJOGo7RWcCurHUCL0KtaHJOeo/dFWjTHG5fzprraaU2x2B8T5fIhTjyu9UlrsOc2rg1qnzEIqf11UdK4LZKiGKo9q1pv0dqrSfK8Swgpv9oKr10RZY4vsA+dNItj0HluawHsqOuX1Vhcruh45VrvgUF41gNvk2X3FssHGK3PEsTLNR8U3wR4m2hvL8gJvj8/Ysv8eVLbgRb3DS4UyWcf1dzAj1InH9Bb+hY3wT5VGjKL1OrhWXHxHRYq/sIWsQ9ZCmMNKv6tuKKE0zBrJDJCaeFFRuZa12Y2ieyWiEeIClbGGXQbMbtQ/CKdTDg9plBJkZhEqxu41P8eoQksgIBRYmfRa5bRwqXtI21u46AfPSHCs/thsrNPDje1wziXyuTpQWdDd1KU5WGGLJTjbBsOdWxHZTaH5Imn+Mldr7Vhk+/byX9Ue/VOR8vdmC3aWNpweyuuHqrVJHTJMIQnWVapoY8yF1b3Lh7dNTWgvUdZY2+d/OgGq8uzADxEycTahiOn/d5fqWou1GsHjWU7lnAQT5Il5JWqmV9pEis90nkcVNuNBCJvoIMxv3LO1++S0OHUXCMESBacn066LeTBXWSstpXqrwkf7H7lYsCKjYGR1vZyH8FLchOPb2TYEO5Vs9vZXnACPjQvVTgA14MJqoI/kDuzHC7jwUEJCcp6xe3BB63fhtj1aERHDvD1tLj/TZ2pd67FZbB8UyQEHHESG27EXV5wwMZqyuNlSVTMctmWLrG4dONukvdlmZFQ3DVVwbt7NkD7Iy0agN/ppMzlYNrFRh91Br8mbos1kEgOLhIFmch2QISsdDCmThi5IyE1d6ag+ulPUvFlAnTV4gVwH2e7DuuxpoyMRhycLKyfv7/qUfuscXBX03bhaWVrq0p7ZCpD1ZgmKbG0o19gXEfVhje1KTwIycrTO+WgU2s7ja1pMt4RD/Wqd0HLc0/9pXn/6WMma+Iml9Fw1ATsaVn2SSnqBIzzn3kAKR+VL53ZMJkEHuZJeY2AHEFJW+0UpM+wK4OiM+PvHCMVbfrp5eZ5nwSEjo3W+VrJSf9qQ5WFhWUd+NVe0h8GSv+vgovcsBxsAoRan2raanzHSvVkU1hpOS/ZK4T27Cvxdnbi8awBDMJPTyQ53+IBh7kZGJaMZPwftHnU1gWSs72mvWLCyIFBCFVuPcFyC5F0MJwIG2S6NvfXV2HzFsdq4Reor5UeJYus5KsjbqWC0hskGidQxxceIDw4uItherKbQUuyz1FNA2CN5qKKu6rhbTOKqBb/V17Jtu70WkzI8qey6llF2tm+0WdvduhcvR/I77BrsI5ArnNuQnWJrANmyXObM+lCnulz0+biX7MZtkCNZHzbIswCawiRz6ZrJHVEAqsJVazd4QG1URrKsSFGLk7tsxuDfZt4YGCfEFTJ3dwRJ0r5+34gugITd7Md/7S9Vi3Qr03oSRdmcHCJ660FeWZ4afzx2wKclpQQsoSnpDu4YkIFb9fmBaQLEy5++XUOD+AXDZdmN0F2ezRBJX6d9VPL9NVExgCVd5NMT1Hq/PsbJB8wDtE+QUTGftE9MpndnGY4VoZX+5yxIe/fTgbVslbJfttC+Rkbp85bql5W0KLJoovuzXXyrZwgEIxthSoVjOEyC7vtVa1BhX6aO8FNDQh3tCkfe4ML8LQN+mHvWg1Lc6kpf6ZdSHxe1nDDQharfE11miQqklugyLK9ztbEeX5G5HQbCDuGq9hlL7QjKRLq/qEkoszur2US3Olj9+TOKTii7ZgqknspoHA2SSyYReAllmkxIk/CZfRcRqEAxZaklwcQvOhfC3E12X3PshD0G8zOrv9RorlxXsWHW7ouinmlvVZ79VpnSUgFd92LiSbb3dtuHtxDgwNte1ej3YLTPYXsuNJa1Mb6zb7qOyzLi1iUrT9yEbM9Bjy1xFmWeJLgtU2o+TzH4K8PuUsU7Jsiv74Y8fRoCcKjbaF3H5yLFclGQfCg4GagxRpdDexfH48VWF+8H1dVbwuLFNvY0VsB3dAwAdts2ynAlKlw1URrsltozOTqtO6EHPfnU1uQBHgFL19jlb4tvdL36ZlvuJfl5SQ5BmqzRfuUXU4nFoqStPmPLpjfaUQ4OZ4EGdLVtQ1YB25Kb9E3sEodeoipBfnfuhRN6Uc++gjd8mobizL4a/BLt8Fkk0JpcQxl4qmO/Xj/LAO9iRDPpXZfK9yIKbOtKVfQWBWZA2MwSbQhzK+NsnWgtmVZtWFY2Fb5C1es2qft7hDg/rjAQSu2zU/N3GC7LCiJ/HzdFZmDtXb5ybJHierv6hZQVDusFJ7BaQZ2riZaOus7WhIfNbpVGrtbJGw+GXhj9VlGIxHq0z9bLWyBVDUsuOgepGKjlll6MwYBQhMAUZx1FJjIKRd7assWjxJbw1BRB3yjhrGElE2bUKIIdIY/WqN7Y/a6iVPjJlEr1EIErK+NNTbGrtWIf0v21CJ5cc+kezxBrlfqzTS18NzJAAI7Qd9FQSChBTLmWfkFbol7FS6Zf1VuqHKCCd9VH+hD1FrmwcypwI4vRYrCMwx5TVfu0s6wSkPgFdf334onHvRbA+S1EZtl+f1S/F229R1jovWkFMT3xSulCC0r5+nH16kTU0MrCcNZl6z0KTrFN2gynlIZxmkl9ln92qkWpy2yyLOktMTezDSXiaB6BLmUlItqoJW5q2nYXJw6HE4ngnybwSM4pS7RYSbBBuApJo7apVmW5vUYMXJ2oopUJ/DVhAq2Bs9VeVVJiNUX1YS3pZUCufujtlCKYysCbERsiMRvB5V8FVGu0ay24q1sTipY5+k6ZEXEalDSnPop2JgtoilpPWc2XnbNinfpuT30RxMfXXzUtAtsEGX8gAsSQpMkIi81Mxnw3Sh7yL0G1is9Z5/piSEYXMm0TUj24vmJJUKmTUZqsSY4P6ybhs7q9Akr8v3QnA0jOIpm3sZIDzrzQJ3MuToBnA5oCSd6C1OFThTxluLeZkG+1RGKFrAVxQxSCEzO002RC+g80/qdCJyHTdGqsLH7VJ0wd97EwqhKHeCxTi1mVsUhGZ8lPqrDDF6PN3ZTSbDJpBffZGN0mhuuoajVlDD28DAD8fbZtwRFVCKqaeTdLBofvA9rcQjMykc1wsyRZr2s5K9+vqA0pJ8Cpj2mJEflQ+L3nCj4v8tnqLIXrtaDDl/UtSjlV1DmVFHvv06PqWeRkm6JgRpmSljYlGBYDzU/13pzSyBfzZ2h+F2tz81ajPtnXgacoRbBk50kGWug9VRlx13HHxbNMVHujD8UWgCPgb3e65jzzVMuul+1CFXKue2a/JopEN3mLzUogmWvvr3ODP1FCdHUSvAtPb26/hrx4x28qcQvf8DO0/F5Y0rqaS7VEXSbBTZamO/OGsHYEphoaOEjxdVRljdbaX9AKnf4t37qR1bpeqQV3J4ldQ2TZJzVshSky1SUpQrq6EsU0gDLSdE/X5fqttvKaJQ0k7P1lB27uuxWFd2gv11+dLfVFyRvbpASa971IHbMcXD2xN8oY80XyS6icbtmgrJD1xpE9qDM/Y29V/uBvCrJdWFoRvLr1wiY1R+uRmdAk7bk8rE56agLgigWZMCTa0FYLFvi6ieK/GC/38qiqqf2kPMkwkG4UmzXZlhsU3YAWTOHGn2i2n3xi0x5Luixoq686JSdiRJ0ps65hqEiKS1tu7S8tn3c4Ts1OFNMOu9ns5gimcKBUMMxURUW4nqCsrwU7decV8rK7xmKQqu7z+rlJGbSpnjQtyUrY5/VM6DGrWrl93Ca4BQQ3GnXUzCPUl7D51lkNnFst8TuEh821uVGdBzBsTP99yJ+NLyrriHie8CUEdb6SBDtVH7ffWx+3Trp4Eu35bpV3F+VN7wMndX5VIdylKNhkASo1zKMUMkCcoO94KF7xa1SqidBbno3/LC57XvN/27VkOEfVqlBFODGNzr76oWPF0xAbhVDk4SxJonIfM4Gp2dZsHtEKr+sHAxn8nf2VnlkAz+HNk7trVvfbbAGdq7EBHTYgvPwSpzwun6AUkl+m8vvBsKbsVeXV/TuxhtrOMgt4SjJ1qwGpfgnl2GEGiqreRgi2Vikhl2fWmbKAjybIVK4v9Ap28o0p+TpRr+8pUWbpjZFfpXzAwJ4HxVFONi+X7g2MJXg4geu9QA75V+MWTUjIUrgJFSIOyugkvh8SC+oVcoStrSE+hAk2hcxbrk6K4rklwNAqUDGT1NFipUHSmvhLFsZsioalQH8QgX8xyly/vouIV2ciCNsTA7VnVTNTFImuvS5y0OHULI1YnqqdVmnY0fVygIFEllg5//2LF3PspsETe2u1S0X1wUyxowZKLwxT5QbzVXyskZsiCua4sdA4lrIzgqBa7uv38Fge9MZIIL+vQsOamluJvWrG3RQFPsDA32WCXYbeiPoUYH6Tr7bzustTtWtHrY6gByTa3wbtLCFuuElyVNJM3dl17nYpTN8DZD9csuO4TxZWGvotSehTsidSqX4/qOGfv2BGjcl7d8opMvKspFJHgrEs5mjj6ak7X0F739QESCZMgWwgQOpfV613iICD+qMYx6Ju3uvfqyjjPNg5UzrwxhGw32ujfh8g7LwpZiEVTkZ6tJ+92L2ssPT6X3gT8HSJj0zEeXxizONDe2eeOTsNMykPKaa52kn4bOrx71ZlI9GDfosjKaKLSI1mHjk1J8me5eEVu05z6qheNCgw7SvfF7Nz3zoDiX/toOLMk7bAzWFHBLpxLULwPIk/oS0oxK93K8j63xlZ3kvjgrNRFI86qHRZ6knmrR7KUxgIP9LH4NcVWKyvrY5FQPSRIVlfw+YvFJXlYH4weFkiPo3BTqA5X7lY7f9jXTTyDCOeQ5pIBYt6qF/gQU5YadAyvGTaO+hBJzML1BHcc6/t+d5xiYR/qkWtsb/SSLLcr6h8qqvhkKI1qkY2XNK78Q3hXWcknjSoksba6uyYF+JAM8mIR7o45qn2H12905TdbqimfCzJ98mwvcPQUDIaof7xVrUILn7r5sHF3kFSOufK6WofzPnfdlIBvinZllpeaIu7mnPYpflR4PcXpRZ2oFmgKL5pE4LxdFe7LTF/1noe6gEfL7soFuoT1Ej+riVJsdyiJFSXtLlWOBDfKNC0FDLrPYh3dXjxFbMrsXN+/epFVENRcXlnTFZn7YsC+KDSGIoS3ulWtXaWrc7unfDFuwEXDxovVhCxCw4PGWU9lLGG8sA0glzgd48M+5AIsWgynGC/uaykgqiC78k1LF2mdoG7f1L24ioFXneTltfuzxZK9PkePZX5b3WO+qGsHJWYDGrekRzJEqioo0FfU5cDt9orT2Rz1wZZcqW82F26Y7a/eMTKmSGdVuca3BOv3p7HeRWxakpqr9mLXjdHwed7LkvYuHQKx2vllhUY0qpdUrVAchMZeQy7sJrdFqfv6Yqq3T/w0633VcRi6DOteGdeFxp57uZFvOqgl6EiXZcoGnNvx4JsLRiGExYLByMYfv50UJUjtmyypdViKsvU7MBiCLRpKO4Is1ztv7Ntfd3WOjzexVpQhWqvPTTlbhNES73JjQsHD1dnPw2aFI9nUG02YPDe2n310k6dozete0us2C2fo2LN1FRNnZvLIaqjuA8MyZMGirJbozVRj1RuoTUrNq3khQCiKHF/XFrhc03+gNdHR02HSuvGqxByxpb/v3nblACDU92uDXfranyey2wJ9NOwLdYnM+e3GVXHRnZ47JuU17LxuF90kasmj4nQNaOo4kwjr0f0KvHK74DAXX0Jshcnq7pFkhfo0dv28lkmnVZyUfokN/Tb4u5NFCPGeLYkBovysL9+KrWu2QW1c8P/ncrDA7Kj5dDmrxp5lnzcN/2TM9029Lt/l3Uf7Xu9Wl/Vcv6qmjrpVyseVIiBPvd/7j9z/gzIgDhwi039qB8JVUmepf/U6uYQ1d1M2trSVuzz1+WRjh0x3tl1EHSbwIWqXXBjEzyyRv/4pWJsiyYGt4rPegQsFscclb9lvazKMqtT4/BBWAsoUSrK7uopqAVSx9GrpWOz7s3Fvd2vwQsmtEx7surcDaYdJD08KGRWhS5jWz6l4FijXWSqEhvCFChcPDxVIm+HV3163b9ercrZCulqLl6ZY3ZNOkHtTi4IzEHX5CmLocRUk6QfVfyHxIA/pa3dKEog029e9AfrTSuuhxhydrHN0EVaTzh23PYdSZkKaOXUnAwABYRwetHuueq2xbtVD2vIS5OL6hxUDrV+YoSM9GzLoVYl0XS0aqJNGXRg47lUX44KcHuRGorjqYpZOXVlaaZy+XFeAGRMPOZQMXrjy+vHTGBa5MzddBe3x8u9OeYXJoZ2QH0qrOAEsS73l6SlCo8fyYmI2QDiB2yCpyKWgWyyI03hZCsUY56c3eKsYE7er5LT1YvDhnVLMuyBu+6FL7TTRZQHCAQ4PoaC3SU0m9TR7SYiikaZluv3YiXa0GsfjhKrFPCVEwfsHR92YqS9iec4Yf7y8xF16YUxHPTqnsTyCzcxbDbGZu9gS4tbwBXqr69xRhC5SX91YNBl6DI7txWzgpaoU9x5XkG/e3tTTRwEZSzAs9QuImE6oAgRHynp0LtTVIK87gOfN5E9fRUI+cCrjXkRR3WdVNzv9orDQy3dXy/e6tbi6fPt0sSMOFZwV+sV3Cna/yqvGjIQ6iC3WrTQlxsFF6OQ7O9esaH2fwuG+F9udzso3bLUFr0w7dSXLtSh8hdHK/kIIS7TIre7cQXybbzBoFMlCjiIhsiBoJc3NmR2sc0Djca9STe/SQ53speaGGSpsbOTLAYs+r75mud8jpQmcAdaNj9D6rgCoFBnP2CUHM3BTu38pEzBb23WsBmFS3FfMNjnTMPudGt8ZPBvhPC5nykpvkcDB9+NIUhRi6PpHKw2PalUE7KcXoAmjaP0gCXzqQqhrVjhUvR1AIc58iS3u9/6kYuueSbLvWeqtOhdIqwCw6Npn8hprNgBvSmvZjf7ObVg6c37dReOFVVx5HRdp4EwW4p29VS8Rh4cu7B5fwfyZDPOOLg5S67OptwQeZdQX0TGhFmiDHCReVq3OirgE5DNZiQeQtkWVpZ3pFJpuVzb2zFMUaBpggq6uyx10F54eG9X+twT5svnz06d8zfpq1SULt+JxFdOmcH5TPe6s/lI20+pqia32UzW20VYvgra8jgullS20ZlUje33/IupnTtqGUQsakQcbBVo9kx4RPWA2+ZpNYdKqq2p5otYuegHD5PlFBqFmbsWJZ31JAdv+tPHb5UpXW+FZ5D0ASLavzDOugMe1aEvtUpu0Yl1VuyQw3L562g+KEMi851UshPE7Iu586otnt5IP5zFhr8qnFhq1+ps1rwPIJwr5HBebPre+eTYdHVV5CEAIHl8f2WQNvXeC2c0+hqqh84fOlrPV6N4o+SDp8CV5Q20Amum9f+RV625j5I6ltnLThVO9Hc5gWc1FQ9iaNKZoAzb5sbjcc+qeu5DE3gB5bJX93wzVUSCJTpbHpHZm1yIUX/1sNegutqvaujXxUoU1qfMfVeuZMxnrKfPeCTOYBPANJDtxqHdNM4DqD41lVGxjBHOpBz5IWD/qLiW2eh27kTKLi+GkiF2XyJXAY5QNNrpwhIr9Or9dHJs6AkYQDyAo22Dj7SoQRLXUc6GMhkzcTqaABAC4+/VAcYkaRGRgHc+N0Sc0DzmOnaIumHJvSSINRenz8iqP0fIthCWD+uJsKelCtO/6eKsWOFyq4iF1hk/3yVa94zg5zNjQzuhfd89aYdpRT9U9vZ1CdFa3pfvtkjbVTY+r3mziobiYwuqSUU39OAuUMH3H2ah8ENHV1btebt936V4wR2C00qfayXntsj5J8pc4vKdwxym1xG8RuOO6qVzn40wWEE19H6UOcJyy2rjQqjgN605o/5LVPh7FVoFO7HGlhybJBV7cwL10moqxDOR0W+MUQH3f8ty7IECRTe3NuDEkMEsOYy0XXfSruugDvKZn2uPWRs75YjJviZB1XSVlVTnjCh2KwNkhp8b3vVXBRPu2+/PsL5GGyCxctu71k6Axj/BhjXdRAZPWbkp6fdz2MD3Xq7JOWCKUcnQEqb8zo+JJoMponTdB2qs0fbX5eJ5OVHsKcGz1aHb18jWWaDgLwevJ0X1hCjqkU/nJezm3wQgLF9WZIrlxb336qLqfGMV/ZCu1UOw8NQrPuJtjFp9//YAjNdEV5+t2sp2gc4MJt/7ppKO57gyO9PNUm8A9SRdwn6L03Ct9OalBNwBkOHpputqV8YjPQQmC4PD6hxRYSaf1W3coK6XLTKkLtMkArY7u9YMq1omo+BjCMPHJDT7iu9ov0eiwgPW0bySBbOGQ5YvWSTSa6nyfavtuKgpDLHRyKCdlUyGE9O+uFD3qinA9lUJdPtV7sy/gPtQhWtMajWkvDwyvckc30Y37vySFo1HV6iZ9Dbt/Sn5ZgYeeKy2SQ4jwXoEVCv1Au+ZIr7saiVEdhbNf9/qOGnowW9Eki+VXU6KwFH2/zB8bGlBqKT7E50rR6FSnB4jrus4RlVz7GIBCR6P3YgAhNY95JQNHNV9p+0UfBeEBQ1Qj47wqY8W7Cl3UAd1jcinX3fn9Mrmiiz4e7CK1IdKIizGTdWievegSCduE8GfI+EoW7gtHDiET5tQ8UG8mYfzFSoua86IrNgefbFD6s9jvi2i/Iv5AL8eE9kpi+FLu592rhz0Sq75egnCHQNbqJcv0kZas3v8Y/DMLUvUlSAd9z1Lo2CJghS7qKGF/1QaK/vqpt8QYVXJEc7SuSe0EaHYVR7jzxiw1a2YyJtW0us903cpcDEr3LRjBeCmLLGgQc8UtpetEEcCySaz4OGSK5yfpkVYK6QyllpSn20cif9RDeVnv0SXM0bWvQm0C49aRY8j0bXWnrltJ2CU2dV8gPsI8xejel0xpN+EJ4RLoUsjWh7h6Yyj4Vx9/TN1OOW/51G9r6K44X0NH3fvdJFBfrzl0CVwFXTGTikTqfR7npnarUeL0avaOSeIghRh1Qcm4Evl1QS4nVCSKoGglpCAp4lDqWd+uP6bxmp7gBaloqSjAkeSQWcv/YVLgylXfSvj65zbDVlPAoOgMXejAWwpcUiYCcjWUhJE4vIjsxQt99v8VpMRiYomG9/N152R8CVR8je6UP8YF5J3tuZNXNW9grpYYJ43QIiGXN5BHMfyGuvT7l2pqLK47aKK2Pm0lJvXt6hSIRTYOxEi3ui+mMA96tB01lBw/XGlxW78oqH6Kf1zeb/FggaBB3YslQhzlfEy9SqEbRRdUnC//fZU42xeFKRavKAM67Kdutfbr+eat4sYSeywkCDqlTCpdsmt9Ng/6npR88HH1iZiirCs7E7sXivFFvG1qgohbHOVprctEB+//OVvNWibQ6WsLbt6cCczdx9cFd+tztfyud6BqBe/KArF8Uyilyuh+JddiK6VyMtvtiCzOCgvvV9dTrRrvZ25BjLu3LPavqDY2F4uEWf8kSkPIwLqEwFCXO7CJir/IEwfyQWa/3lSY2CborOBvSIw6YxdtAJAoJiVaDFKfvLWxrqr61JHDyM13Cu2+dNumyQVKzUVDeXEyL4KnoMft+o+iIPP7TbIZNwlaul6o/z/Ptu41RIkOITgbVFXKPxfLDuMFSaZbko3Y3BCemdFFScuH6Ya8kMfa6gDbIhR9RUF2BUxxBdQ2uQL/qBOeeqouwTm8tVT3VLCYz6NV3bdh6ozvvP7Y7PLUPoBeuQ1nkR5g+iT3QY7IJHpY1INwMjYrshac1m/CDNo2v8u53Uz1TN1diZ4Vqx2rilI4w0kCpYMz62LznNKq4mpJHGhjJVoR3F1EfDxXcEW4ZDk6s7b9rUG26/ZMbhm/yla8uJCxiC5O3FelLnRLXICZvNixtgX7rWL2a6j4ztClggZUdYKJyFPZijRzvdQ3utpKyKUdX2LLcdpn+rdyoM169pHFMtUJAi0qrobJU6FI/xaqq6tr41Dkglppp1Td9m3w6fdG3Di8f/RUmNEUO7Svq8JFp4pDRj3D8WCV/sMOFp9Lj2WfZNhNRhxCPSpr/XhiD3PhE7zfRK04R2Ln/gle1H50GoUqptqnb0Gv38720A1oaNh2EFAZHZcRqKhR+/gM9pctXfqral6xlOg0on3pxH4ibN0w2GjdOVWiw0POckVlt/MWgOxKekTdec2UNa4u+0ftv98ujwjCCcaMjP6ly32SSn8NfKyCSw+AQF3/8256hQ8HNJ4okLd2bRSLdQ2ISo8VZQaXGzK9ua/pW0Is7s6KGidfaIYDzOuVMsanD69AgOClcxRpmaX/hwC3qOm1UYP8MRClyfgubIdr7yWU9AxmfugMMl76kZJTFfWn/93/Af/KIZbxpgAA"""

def get_training_dataframe():
    for p in [ROOT / "data" / "student_performance.csv", ROOT / "student_performance.csv"]:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    decomp = gzip.decompress(base64.b64decode(EMBEDDED_CSV_DATA))
    return pd.read_csv(io.BytesIO(decomp))

def load_pipeline():
    for mp in [ROOT / "artifacts" / "model_pipeline.joblib", ROOT / "model_pipeline.joblib"]:
        if mp.exists():
            try:
                return joblib.load(mp)
            except Exception:
                pass

    from sklearn.ensemble import HistGradientBoostingRegressor
    try:
        df = get_training_dataframe()
        valid_mask = (df["FinalExamScore"] >= 0.0) & (df["FinalExamScore"] <= 100.0)
        df_clean = df[valid_mask].copy()
        X = df_clean.drop(columns=["FinalExamScore"])
        y = df_clean["FinalExamScore"].values
        champion = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=0.5,
            random_state=42
        )
        pipe = build_pipeline(model=champion)
        pipe.fit(X, y)
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, MODEL_PATH)
        except Exception:
            pass
        return pipe
    except Exception as e:
        print("Model loading error:", e)
        return None

RMSE_ESTIMATE = 6.949
PASSING_THRESHOLD = 50.0

pipeline = load_pipeline()

try:
    import spaces
    HAS_SPACES = True
except Exception:
    HAS_SPACES = False


def predict_student(
    previous_score: float,
    attendance: float,
    assignments: float,
    backlogs: float,
    study_hours: float,
    sleep_hours: float,
    participation: float,
    extracurricular: float
):
    if pipeline is None:
        return (
            "<p style='color: red;'>Error: Model pipeline artifact not found.</p>",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    input_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    raw_pred = pipeline.predict(input_df)[0]
    score = float(np.clip(raw_pred, 0.0, 100.0))

    ci_lower = max(0.0, round(score - 1.96 * RMSE_ESTIMATE, 1))
    ci_upper = min(100.0, round(score + 1.96 * RMSE_ESTIMATE, 1))

    # Calculate probabilities using normal distribution CDF
    z_score = (score - PASSING_THRESHOLD) / RMSE_ESTIMATE
    pass_prob = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    pass_prob = float(np.clip(pass_prob, 0.01, 0.99))
    risk_prob = 1.0 - pass_prob

    if score >= 75.0:
        standing_tier = "High Distinction"
        tier_color = "#1b5e20"
        badge_bg = "#e8f5e9"
        advice = "Excellent academic momentum. Maintain current study patterns and steady sleep schedule."
    elif score >= 50.0:
        standing_tier = "Satisfactory Pass"
        tier_color = "#0d47a1"
        badge_bg = "#e3f2fd"
        advice = "On track to pass successfully. Increasing weekly study hours can push performance to distinction."
    else:
        standing_tier = "At-Risk of Failing"
        tier_color = "#b71c1c"
        badge_bg = "#ffebee"
        advice = "Proactive academic intervention recommended. Prioritize clearing pending backlogs and attending tutorial office hours."

    # Top Cards HTML matching prototype card grid
    cards_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 8px; margin-bottom: 20px;">
        <!-- Card 1: Score -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Score</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #1e293b; margin: 8px 0 4px 0;">{score:.1f} <span style="font-size: 1.1rem; color: #94a3b8; font-weight: 500;">/ 100</span></div>
            <div style="display: inline-block; background: {badge_bg}; color: {tier_color}; padding: 3px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600;">{standing_tier}</div>
        </div>

        <!-- Card 2: Pass Probability -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Pass Probability</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #848408; margin: 8px 0 4px 0;">{pass_prob * 100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.82rem;">Score &ge; 50 threshold</div>
        </div>

        <!-- Card 3: At-Risk Probability -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">At-Risk Probability</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #800020; margin: 8px 0 4px 0;">{risk_prob * 100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.82rem;">Deficiency risk index</div>
        </div>

        <!-- Card 4: 95% Confidence Interval -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">95% Confidence Range</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #1e293b; margin: 8px 0 4px 0;">{ci_lower}% &ndash; {ci_upper}%</div>
            <div style="color: #64748b; font-size: 0.82rem;">Based on &plusmn;1.96 RMSE (6.95)</div>
        </div>
    </div>
    """

    # Bar chart (Olive #848408 & Maroon #800020)
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=180)
    categories = ["Pass Probability", "At-Risk Probability"]
    probabilities = [pass_prob * 100, risk_prob * 100]
    bar_colors = ["#848408", "#800020"]

    bars = ax.bar(categories, probabilities, color=bar_colors, width=0.32)
    ax.set_ylabel("Probability (%)", fontsize=9, color="#555555")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color="#777777", fontsize=8.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, color="#333333", fontsize=9.5, fontweight="600")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#d0d5dd")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333"
        )

    plt.tight_layout()

    # Advisory banner
    advisory_html = f"""
    <div style="background: #f8fafc; border-left: 4px solid {tier_color}; border-radius: 4px; padding: 12px 16px; margin-top: 10px;">
        <span style="font-weight: 600; color: #1e293b;">Advisor Guidance:</span>
        <span style="color: #475569; margin-left: 6px;">{advice}</span>
    </div>
    """

    return cards_html, gr.update(value=fig, visible=True), gr.update(value=advisory_html, visible=True)


if HAS_SPACES:
    predict_student = spaces.GPU(predict_student)


# Custom styling for prototype look:
# Soft pastel slate-blue action button (#e3e8f1), clean inputs, elegant typography
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

#recommend_btn {
    background: #e3e8f1 !important;
    color: #1e293b !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border: 1px solid #ccd5e2 !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    height: 100% !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}

#recommend_btn:hover {
    background: #d4dde9 !important;
    border-color: #b8c4d4 !important;
}

#recommend_btn:active {
    background: #c5d1e0 !important;
}
"""

initial_placeholder = """
<div style="background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 32px 20px; text-align: center; color: #64748b; margin-top: 10px;">
    <div style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
    <div style="font-size: 1.05rem; font-weight: 600; color: #334155;">No predictions generated yet</div>
    <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px;">
        Adjust the student background and study metrics above, then click <strong>Find predictions</strong> to forecast final exam results.
    </div>
</div>
"""

with gr.Blocks(title="Student Performance Predictor", css=custom_css) as demo:
    # 1. Clean left-aligned title matching prototype
    gr.Markdown("# Student Performance Predictor")

    # 2. Controls Row: Inputs on the left, action button on the right
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                prev_score = gr.Slider(0.0, 100.0, value=72.0, step=0.5, label="Previous Exam Score (0–100)")
                attendance = gr.Slider(0.0, 100.0, value=82.0, step=1.0, label="Attendance Percentage (%)")
                assignments = gr.Slider(0.0, 100.0, value=85.0, step=1.0, label="Assignments Completed (%)")
                backlogs = gr.Number(value=0, precision=0, label="Previous Backlogs")

            with gr.Row():
                study_hours = gr.Slider(0.0, 16.0, value=5.0, step=0.5, label="Daily Study Hours")
                sleep_hours = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Daily Sleep Hours")
                participation = gr.Slider(0.0, 10.0, value=6.5, step=0.5, label="Class Participation (0–10)")
                extracurricular = gr.Slider(0.0, 20.0, value=4.0, step=0.5, label="Extracurricular Hours")

        with gr.Column(scale=1, min_width=180):
            predict_btn = gr.Button("Find predictions", elem_id="recommend_btn")

    # 3. Section Title matching prototype "Recommendations" heading
    gr.Markdown("## Recommendations")

    # 4. Result cards & visualizations (Initialized with clean placeholder, no auto-trigger)
    cards_out = gr.HTML(value=initial_placeholder)
    chart_out = gr.Plot(visible=False)
    advisory_out = gr.HTML(visible=False)

    # Triggered ONLY when user explicitly clicks the button
    predict_btn.click(
        fn=predict_student,
        inputs=[
            prev_score, attendance, assignments, backlogs,
            study_hours, sleep_hours, participation, extracurricular
        ],
        outputs=[cards_out, chart_out, advisory_out]
    )

if __name__ == "__main__":
    demo.launch()
