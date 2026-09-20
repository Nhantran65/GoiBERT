"""Heuristic extractor for first-person restaurant/hotel review passages from FineWeb-2 vie_Latn documents.

Plan §3: keep only passages that carry (a) a restaurant/hotel *venue* marker AND (b) a first-person experience /
evaluation expression, and reject room listings, property descriptions, news, ads/SEO, recipes, advice/listicles,
fiction and machine-translated review pages.  Rules are frozen before any downstream number is looked at and were
tuned only by reading samples for precision (the 100-per-cell manual audit lives in prepare_corpus.py).

Unit of extraction: a newline-separated paragraph.  Inside a paragraph we split into sentences, tag each sentence
as review-like / neutral / negative, and keep the longest contiguous run of non-negative sentences that contains at
least one review-like sentence (neutral sentences at the run edges are trimmed).  The kept run must still satisfy
the passage-level gate.  No sentence is rewritten; nothing is generated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


def _rx(words: Iterable[str]) -> re.Pattern:
    alts = sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True)
    body = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in alts)
    return re.compile(r"(?<![\w])(?:" + body + r")(?![\w])", re.I)


# ------------------------------------------------------------------ domain (venue) markers: strong nouns only
HOTEL_VENUE = _rx("""khách sạn|ks|resort|homestay|hostel|nhà nghỉ|guesthouse|motel|lễ tân|nhận phòng|trả phòng|check in|check-in|checkin|check out|check-out|checkout|đặt phòng|book phòng|phòng ốc|phòng nghỉ|phòng deluxe|phòng standard|phòng superior|phòng suite|phòng dorm|dorm|bungalow|lưu trú|nghỉ dưỡng|ở lại đây|ở đây 1 đêm|ở đây 2 đêm|ở đây 3 đêm|1 đêm|2 đêm|3 đêm|4 đêm|5 đêm|một đêm|hai đêm|ba đêm|qua đêm|buồng phòng|housekeeping|bữa sáng|ăn sáng|buffet sáng|bể bơi|hồ bơi|view biển|hướng biển""".split("|"))
REST_VENUE = _rx("""nhà hàng|quán|quán ăn|quán nhậu|quán cafe|quán cà phê|quán ốc|quán lẩu|quán bún|quán phở|quán cơm|quán chè|quán bánh|quán kem|quán trà sữa|quán nướng|quán này|quán đó|quán đây|quán nó|tiệm|tiệm ăn|tiệm bánh|tiệm cơm|tiệm phở|buffet|gọi món|order|thực đơn|menu|đầu bếp|bếp|phục vụ|nước chấm|nước lèo|nước dùng|khẩu vị|hương vị|đồ ăn|thức ăn|món ăn|phần ăn|khẩu phần|suất ăn|ăn ở đây|ăn tại đây|ăn tại|đi ăn|ăn thử|thử món|bàn|bill|hóa đơn|tính tiền|nướng|lẩu|hải sản|cơm|bún|phở|bánh|chè|kem|trà sữa|cà phê|cafe|bia|hủ tiếu|mì|pizza|sushi|bò|gà|tôm|cá|mực|ốc|thịt|rau|canh|súp|salad|ngon|dở""".split("|"))
STRONG_HOTEL = _rx("khách sạn|ks|resort|homestay|hostel|nhà nghỉ|guesthouse|motel|lễ tân|nhận phòng|trả phòng|check in|check-in|checkin|check out|check-out|checkout|đặt phòng|book phòng|phòng ốc|phòng nghỉ|phòng deluxe|phòng standard|phòng superior|phòng suite|phòng dorm|dorm|bungalow|lưu trú|nghỉ dưỡng|buồng phòng|housekeeping|bữa sáng|buffet sáng|bể bơi|hồ bơi".split("|"))
STRONG_REST = _rx("nhà hàng|quán|quán ăn|quán nhậu|quán cafe|quán cà phê|quán ốc|quán lẩu|quán bún|quán phở|quán cơm|quán chè|quán bánh|quán kem|quán trà sữa|quán nướng|tiệm|tiệm ăn|tiệm bánh|tiệm cơm|tiệm phở|buffet|gọi món|order|thực đơn|menu|đầu bếp|nước chấm|nước lèo|nước dùng|đồ ăn|thức ăn|món ăn|phần ăn|khẩu phần|suất ăn|ăn ở đây|ăn tại đây|đi ăn|ăn thử|thử món|bill|hóa đơn|tính tiền".split("|"))

# ------------------------------------------------------------------ user voice
FIRST_PERSON = _rx("""mình|tui|tôi|tớ|tao|bọn mình|tụi mình|chúng mình|bọn tớ|tụi tớ|bọn tui|tụi tui|bọn tao|tụi tao|nhà mình|nhà tui|nhà tôi|gia đình mình|gia đình tôi|gia đình tui|vợ chồng mình|vợ chồng tôi|vợ chồng tui|vợ mình|chồng mình|ck mình|vk mình|bạn mình|nhóm mình|cả nhà mình|con mình|bé nhà mình|mẹ mình|ba mình|bố mình|ba mẹ mình|bố mẹ mình|em và|em đi|em ăn|em ở|em thấy|em thì|em đặt|em gọi|em order|em book|em ghé|em đến|em tới|em có|em đã|em từng|em mới|em vừa|em qua|em vào|em ngồi|em ngủ|e đi|e ăn|e ở|e thấy|e đặt|e gọi|e book|e ghé|e đến|e có|e đã|e mới|e vừa|m đi|m ăn|m ở|m thấy|m đã|m mới|t đi|t ăn|t ở|t thấy|t đã|t mới""".split("|"))
REFLEXIVE = re.compile(r"(?<![\w])(?:của|cho|bản thân|tự|chính|giữ|hại|bảo vệ|thân|riêng|với|về|đến|tới|lấy|tìm|xem|nghĩ|nhận|biết|coi|thấy)\s+mình(?![\w])", re.I)
ADVICE = _rx("""bạn|các bạn|bạn nào|quý bạn|quý vị|quý khách|du khách|thực khách|khách hàng|người dùng|mọi người nên|chúng ta|ta|ai đã|những ai|ai muốn|nếu muốn|hãy|nên|cần|đừng|không nên|ko nên|sẽ giúp|giúp bạn|cho bạn|dành cho|phù hợp với|phù hợp cho|thích hợp cho""".split("|"))

EXPERIENCE = _rx("""đã ăn|đã ở|đã đến|đã tới|đã ghé|đã thử|đã đặt|đã book|đã dùng|đã nghỉ|đã từng|đã đi|đã qua|đã ngủ|từng ăn|từng ở|từng đến|từng ghé|từng đi|có ăn|có ở|có ghé|có đến|có thử|có đặt|có book|có đi|hay ăn|hay ghé|hay đến|hay đi|hay ở|thường ăn|thường ghé|thường đến|thường đi|mới ăn|mới ở|mới ghé|mới đi|mới đến|mới thử|mới đặt|mới book|vừa ăn|vừa ở|vừa ghé|vừa đi|vừa đến|vừa thử|vừa đặt|vừa book|đi ăn|đi ở|đi ghé|đi thử|ghé|ghé qua|ghé ăn|ghé quán|ghé lại|ghé vào|ăn thử|thử món|gọi món|gọi|order|đặt phòng|book phòng|đặt bàn|book bàn|book|nghỉ ở|ở đây|ở lại|ở tại|ở khách sạn|ở resort|ở homestay|ở nhà nghỉ|ở ks|ở 1 đêm|ở 2 đêm|ở 3 đêm|đến đây|tới đây|vào đây|qua đây|ăn ở đây|ăn tại|ăn tại đây|quay lại|quay lại lần sau|trở lại|lần đầu|lần đầu tiên|lần này|lần trước|lần sau|lần 2|lần 3|lần thứ|hôm nay|hôm qua|hôm trước|hôm bữa|hôm nọ|hôm đó|hôm rồi|bữa nay|bữa trước|bữa đó|bữa rồi|tối qua|tối nay|tối hôm|trưa nay|trưa qua|trưa hôm|sáng nay|sáng qua|sáng hôm|chiều nay|chiều qua|chiều hôm|cuối tuần rồi|cuối tuần vừa rồi|cuối tuần trước|tuần trước|tuần rồi|tháng trước|tháng rồi|đợt rồi|đợt trước|đợt vừa rồi|đợt này|hồi|dịp|nhân dịp|kỳ nghỉ|chuyến đi|chuyến này|đi du lịch|đi công tác|đi chơi|đi phượt|ngồi ở đây|ngồi quán|vô quán|vào quán|tới quán|đến quán|qua quán|tới nhà hàng|đến nhà hàng|vào nhà hàng|tới khách sạn|đến khách sạn|vào khách sạn|nhận phòng|trả phòng|check in|check-in|checkin|check out|check-out|checkout|ship|đặt ship|gọi ship|đặt về|mua về|mang về|đem về|ăn xong|ăn hết|ăn no|ăn rồi""".split("|"))

EVALUATION = _rx("""ngon|ngon lắm|ngon tuyệt|rất ngon|khá ngon|ngon miệng|ngon xỉu|dở|dở tệ|tệ|quá tệ|tệ hại|tuyệt|tuyệt vời|xuất sắc|hoàn hảo|ok|okay|oki|okie|ổn|ổn áp|tạm|tạm được|tạm ổn|bình thường|bt|cũng được|không tệ|ko tệ|sạch|sạch sẽ|siêu sạch|bẩn|dơ|cũ|xuống cấp|ẩm|mốc|hôi|thân thiện|nhiệt tình|chu đáo|tận tình|dễ thương|niềm nở|vui vẻ|lịch sự|thái độ|khó chịu|cau có|thờ ơ|chậm|lâu|chờ lâu|đợi lâu|nhanh|đắt|mắc|rẻ|hợp lý|phải chăng|bình dân|chát|hơi đắt|hơi mắc|đáng tiền|đáng đồng tiền|xứng đáng|không đáng|ko đáng|rộng|rộng rãi|chật|chật chội|đẹp|xấu|thoáng|thoáng mát|thoải mái|dễ chịu|ồn|ồn ào|yên tĩnh|thất vọng|hài lòng|ưng|ưng ý|thích|rất thích|cực thích|mê|ghiền|nghiện|điểm trừ|điểm cộng|trừ điểm|cộng điểm|recommend|rcm|nên thử|nên ghé|đáng thử|đáng ghé|khuyên|sẽ quay lại|sẽ ghé lại|sẽ trở lại|không bao giờ quay lại|ko bao giờ quay lại|ủng hộ|ủng hộ dài dài|10 điểm|9 điểm|8 điểm|5 sao|4 sao|3 sao|2 sao|1 sao|mặn|nhạt|lạt|cay|ngọt|béo|ngấy|tanh|dai|bở|giòn|mềm|tươi|khô|nguội|nóng hổi|vừa miệng|hợp khẩu vị|đậm đà|nhạt nhẽo|đa dạng|phong phú|ít|nhiều|đầy đủ|thiếu|hỏng|hư|kém|tốt|rất tốt|khá tốt|quá tốt|tuyệt cú mèo|hết sảy|số dách|bao ngon|bao rẻ|bao sạch|ấm cúng|xinh|xinh xắn|chill|sang|sang chảnh|hơi|khá|rất|cực|quá|siêu|lắm|ghê|dữ""".split("|"))

REVIEW_NOUNS = _rx("""nhân viên|nv|phục vụ|thái độ|chủ quán|chủ nhà|chủ ks|anh chủ|chị chủ|cô chủ|bà chủ|ông chủ|lễ tân|bảo vệ|giữ xe|chỗ để xe|chỗ đậu xe|bãi xe|không gian|không khí|view|khung cảnh|decor|trang trí|giá|giá cả|giá tiền|tiền|k|vnd|vnđ|bill|hóa đơn|đồ ăn|thức ăn|món|món ăn|khẩu phần|phần ăn|suất|đĩa|dĩa|tô|ly|bát|chén|phòng|giường|nệm|gối|chăn|ga|toilet|wc|nhà vệ sinh|phòng tắm|nước nóng|vòi sen|bồn tắm|máy lạnh|điều hòa|điều hoà|quạt|wifi|internet|tivi|tv|tủ lạnh|bữa sáng|ăn sáng|buffet sáng|hồ bơi|bể bơi|thang máy|ban công|cửa sổ|vị trí|địa điểm|đường đi|chỗ ngồi|bàn ghế|bàn|ghế|vệ sinh|khăn|dép|đồ dùng|tiện nghi|cách âm|mùi|nước chấm|nước lèo|nước dùng|nước sốt|topping|rau|thịt|bánh|cơm|bún|phở|lẩu|nướng|hải sản|đồ uống|nước uống|trà|cà phê|cafe|bia|menu|thực đơn|order|check in|check out|nhận phòng|trả phòng|khách|đông|vắng|đợi|chờ|giao hàng|ship|shipper|đóng gói|hộp""".split("|"))

COLLOQUIAL = _rx("""nhé|nha|nhá|nhen|nghen|nè|nà|á|ạ|ơi|thui|hok|hông|hem|ko|kg|hổng|dc|đc|đk|ntn|mn|ae|ace|cả nhà|hehe|hihi|haha|hjhj|kkk|kaka|=))|:))|:)|:d|:v|^^|<3|lun|luôn|lắm|quá trời|quá chừng|xỉu|siêu|cực|cực kỳ|cực kì|vãi|đỉnh|xịn|xịn xò|ok la|okla|ưng|ưng cái bụng|mê|ghiền|iu|vote|rate|10/10|9/10|8/10|7/10|6/10|5/10|10đ|9đ|8đ|7đ""".split("|"))

# ------------------------------------------------------------------ negatives
BUSINESS = _rx("""quý khách|quý vị|quý khách hàng|khách hàng thân thiết|chúng tôi|liên hệ|hotline|đường dây nóng|đặt ngay|đặt phòng ngay|đặt bàn ngay|book ngay|gọi ngay|mua ngay|đăng ký ngay|ưu đãi|khuyến mãi|khuyến mại|giảm giá|giá chỉ|chỉ từ|chỉ với|combo|voucher|mã giảm|miễn phí vận chuyển|freeship|giao hàng tận nơi|nhận đặt|nhận order|nhận ship|nhận giao|đặt hàng|website|web|trang web|fanpage|facebook|fb|zalo|inbox|ib|sđt|sdt|số điện thoại|điện thoại|email|mail|địa chỉ liên hệ|cam kết|bảo hành|uy tín|chuyên nghiệp|giá rẻ nhất|tốt nhất|hàng đầu|số 1|top 1|tọa lạc|toạ lạc|nằm tại|nằm ở|nằm ngay|nằm trên|nằm trong|nằm cách|sở hữu|cung cấp|trang bị|được trang bị|được thiết kế|được xây dựng|được bố trí|được đánh giá|được biết đến|được mệnh danh|được xem là|được coi là|được nhiều|bao gồm|với hệ thống|với đội ngũ|đội ngũ|mang đến|đem đến|mang lại|đem lại|du khách|thực khách|khách du lịch|khách tham quan|khách lưu trú|khách hàng|người dùng|người tiêu dùng|tiêu chuẩn|đạt chuẩn|đẳng cấp|thượng lưu|lý tưởng|hoàn hảo cho|điểm đến|thiên đường|nổi tiếng|nức tiếng|trứ danh|hàng nghìn|hàng ngàn|hàng trăm|toàn quốc|cả nước|khắp nơi|mọi miền|giới thiệu|xin giới thiệu|gợi ý|tổng hợp|danh sách|top|bỏ túi|lưu lại|note lại|cẩm nang|kinh nghiệm|hướng dẫn|bí kíp|bí quyết|mẹo|tips|tham khảo|nguồn|theo|sản phẩm|sản xuất|thiết kế|mẫu|chất liệu|công nghệ|tính năng|thiết bị|máy|dịch vụ|kinh doanh|đầu tư|mở quán|mở nhà hàng|chủ đầu tư|doanh thu|lợi nhuận|khách hàng|thị trường|marketing|quảng cáo|thương hiệu|chuỗi|franchise|nhượng quyền|tuyển|tuyển dụng|việc làm|lương|nhân sự|ứng viên|cv""".split("|"))
NEWS = _rx("""cho biết|cho hay|chia sẻ với|trao đổi với|trả lời|phóng viên|pv|báo chí|tòa soạn|toà soạn|ttxvn|vnexpress|tuổi trẻ|thanh niên|dân trí|vietnamnet|zing|kenh14|ubnd|hđnd|sở|bộ|công an|cơ quan|chính quyền|thanh tra|kiểm tra|xử phạt|vi phạm|quy định|nghị định|thông tư|luật|dự án|doanh nghiệp|công ty|tập đoàn|tổng công ty|giám đốc|chủ tịch|phó chủ tịch|ông|bà|anh ấy|chị ấy|kỷ niệm|hội nghị|hội thảo|khai trương|khai mạc|khánh thành|tăng trưởng|tỷ đồng|triệu đồng|nghìn tỷ|ngàn tỷ|phát biểu|phát động|tổ chức|thống kê|tỷ lệ|phần trăm|nghiên cứu|chuyên gia|bác sĩ|bệnh|bệnh nhân|điều trị|sức khỏe|sức khoẻ|dinh dưỡng|chế độ ăn|thực phẩm|calo|vitamin|giảm cân|tăng cân|huyết áp|tim mạch|răng|da|mụn|tóc|thuốc""".split("|"))
FICTION = _rx("""nàng|chàng|hắn|gã|y|thị|lão|mụ|cô ấy|anh ấy|chị ấy|ông ta|bà ta|cô ta|anh ta|hắn ta|nó|chúng nó|bọn họ|họ|người ta|thì thầm|gật đầu|lắc đầu|mỉm cười|cười khẩy|cười nhạt|khẽ|bỗng|chợt|bất giác|ngước|liếc|nhíu mày|cau mày|thở dài|nước mắt|run rẩy|gào|hét|thét|quát|siết|ôm|hôn|nắm tay|buông|vội vã|lặng lẽ|lẩm bẩm|trầm ngâm|ánh mắt|đôi mắt|khuôn mặt|gương mặt|bờ môi|mái tóc|bóng dáng|căn phòng|cánh cửa|bước vào|bước ra|quay người|quay đầu|chương|hồi|tập|full|truyện|tiểu thuyết|ngôn tình|kiếm hiệp|xuyên không|tu tiên|linh khí|tiên tôn|đại nhân|công tử|tiểu thư|thiếu gia|sư phụ|sư huynh|sư muội|đệ tử|bệ hạ|hoàng thượng|nương nương|ngươi|các ngươi|huynh|muội|đệ|tỷ|nói:|hỏi:|đáp:|bảo:|cười:|“|”""".split("|"))
RECIPE = _rx("""cách làm|cách nấu|cách chế biến|cách pha|công thức|nguyên liệu|chuẩn bị|sơ chế|bước 1|bước 2|bước 3|bước 4|bước 5|cho vào|đun|đun sôi|nấu chín|xào|chiên|rán|hấp|luộc|ướp|thái|băm|xay|trộn đều|nêm|nêm nếm|gia vị|thìa|muỗng|gram|gr|kg|ml|lít|độ c|lửa nhỏ|lửa lớn|lửa vừa|làm tại nhà|tại nhà|tự làm|homemade""".split("|"))

MT_PAGE = re.compile(r"dịch máy|bản dịch|Google cung cấp|dịch tự động|được dịch từ|Xem bản gốc", re.I)
STORY_HOST = re.compile(r"truyen|story|fanfic|wattpad|ngontinh|tieuthuyet|doctruyen|sachvui|gacsach|webtruyen|vozer|voz\b", re.I)
EXCLUDED_HOSTS = ("tripadvisor.",)   # machine-translated reviews dominate the Vietnamese locale pages
PHONE = re.compile(r"(?<!\d)(?:\+84|0)\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}(?!\d)")
URL = re.compile(r"https?://|www\.|\.com\b|\.vn\b|\.net\b|\.org\b", re.I)
PRICE_LIST = re.compile(r"(?:\d[\d.,]*\s*(?:đ|vnđ|vnd)\b[^.]{0,40}){3,}|(?:giá phòng|loại phòng|hạng phòng|bảng giá|giá bán|giá thuê|giá tour|giá vé)", re.I)
LIST_ITEM = re.compile(r"^\s*(?:[-–•*+]|\d+[.)]|[a-z][.)])\s+", re.I)
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[^\s])|(?<=[.!?…])\s*(?=[A-ZÀ-Ỹ])|\s*[\r\n]+\s*")
WORD = re.compile(r"[^\W_]+", re.U)
VI_CHARS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")
VI_COMMON = set("và của là có không được cho với các những này để trong một mình rất cũng nhưng tôi thì nhân viên phòng ăn ngon khách sạn quán đẹp giá tốt sạch sẽ nên lại đã bị hơi ở về từ ra vào khá lắm quá ok ko dc nhé nhiều còn khi đến mà chỉ thấy đi vì nè nha".split())

MIN_WORDS = 12
MAX_WORDS = 320
DOC_FICTION_PER_100W = 1.5


@dataclass
class Passage:
    text: str
    domain: str          # hotel | restaurant
    n_words: int
    first_person: int
    evaluation: int
    review_nouns: int
    colloquial: int
    experience: int
    kept_sentences: int
    total_sentences: int


def is_vietnamese(text: str) -> bool:
    if CJK.search(text):
        return False
    words = [w.lower() for w in WORD.findall(text)]
    if not words:
        return False
    dia = len(VI_CHARS.findall(text)) / max(1, len(text))
    common = sum(w in VI_COMMON for w in words) / len(words)
    return dia >= 0.02 or common >= 0.15


def split_sentences(paragraph: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(paragraph) if p and p.strip()]
    return parts or [paragraph.strip()]


def first_person_count(s: str) -> int:
    return max(0, len(FIRST_PERSON.findall(s)) - len(REFLEXIVE.findall(s)))


def sentence_tag(s: str, tier: str = "strict") -> str:
    """'neg' = business/news/recipe/fiction/advice/price/phone/url voice; 'rev' = first-person user voice with
    evaluation or experience content; 'neu' otherwise.  tier='relaxed' is the documented upper-bound variant
    (diagnostic only): it also accepts evaluative sentences without a first-person cue."""
    if PHONE.search(s) or URL.search(s) or PRICE_LIST.search(s):
        return "neg"
    fp = first_person_count(s)
    ev = len(EVALUATION.findall(s))
    exp = len(EXPERIENCE.findall(s))
    nouns = len(REVIEW_NOUNS.findall(s))
    col = len(COLLOQUIAL.findall(s))
    neg = len(BUSINESS.findall(s)) + len(NEWS.findall(s)) + len(RECIPE.findall(s))
    fic = len(FICTION.findall(s))
    adv = len(ADVICE.findall(s))
    if neg >= 1:
        return "neg"
    if fic >= 1 and fp == 0:
        return "neg"
    if fic >= 2:
        return "neg"
    if adv >= 1 and fp == 0:
        return "neg"
    if fp >= 1 and (ev >= 1 or exp >= 1) and (nouns >= 1 or col >= 1 or exp >= 1):
        return "rev"
    if fp == 0 and ev >= 2 and nouns >= 1 and (col >= 1 or exp >= 1) and adv == 0:
        return "rev"          # e.g. "Đồ ăn ngon, nhân viên nhiệt tình, sẽ quay lại nhé"
    if tier == "relaxed" and fp == 0 and ev >= 2 and nouns >= 1 and adv == 0:
        return "rev"
    return "neu"


def domain_of(text: str) -> str | None:
    sh, sr = len(STRONG_HOTEL.findall(text)), len(STRONG_REST.findall(text))
    if not (sh or sr):
        return None
    if sh and not sr:
        return "hotel"
    if sr and not sh:
        return "restaurant"
    h, r = len(HOTEL_VENUE.findall(text)), len(REST_VENUE.findall(text))
    return "hotel" if (sh, h) >= (sr, r) else "restaurant"


def document_rejected(document: str, url: str) -> str | None:
    host = urlparse(url or "").netloc.lower()
    if any(x in host for x in EXCLUDED_HOSTS):
        return "excluded_host"
    if STORY_HOST.search(host) or STORY_HOST.search(urlparse(url or "").path.lower()):
        return "story_host"
    if MT_PAGE.search(document or ""):
        return "mt_page"
    words = len(WORD.findall(document or ""))
    if words and len(FICTION.findall(document)) * 100.0 / words >= DOC_FICTION_PER_100W:
        return "fiction_doc"
    return None


def extract_passages(document: str, url: str, *, normalize, tier: str = "strict") -> tuple[list[Passage], dict[str, int]]:
    """Return passages found in one document plus per-document rejection counters.

    tier='strict' is the plan's filter (venue marker + non-reflexive first person + experience cue + >=2 evaluations
    + >=2 review nouns).  tier='relaxed' is a diagnostic upper bound: first person OR colloquial cue, experience cue
    OR >=3 evaluations, >=1 review noun.  Both share the same negatives and the same sentence-run trimming."""
    stats = {"docs": 1, "paragraphs": 0, "too_short": 0, "not_vi": 0, "no_domain": 0, "no_review_run": 0, "gate_fail": 0, "kept": 0}
    reason = document_rejected(document, url)
    if reason:
        stats[reason] = 1
        return [], stats
    passages: list[Passage] = []
    for raw in (document or "").split("\n"):
        para = normalize(raw)
        para = LIST_ITEM.sub("", para)
        stats["paragraphs"] += 1
        if len(WORD.findall(para)) < MIN_WORDS:
            stats["too_short"] += 1; continue
        if not is_vietnamese(para):
            stats["not_vi"] += 1; continue
        if domain_of(para) is None:
            stats["no_domain"] += 1; continue
        sents = split_sentences(para)
        tags = [sentence_tag(s, tier) for s in sents]
        best: tuple[int, int] | None = None
        i = 0
        while i < len(sents):
            if tags[i] == "neg":
                i += 1; continue
            j = i
            while j < len(sents) and tags[j] != "neg":
                j += 1
            revs = [k for k in range(i, j) if tags[k] == "rev"]
            if revs:
                a, b = revs[0], revs[-1] + 1
                if best is None or (b - a) > (best[1] - best[0]):
                    best = (a, b)
            i = j
        if best is None:
            stats["no_review_run"] += 1; continue
        text = " ".join(sents[best[0]:best[1]])
        n_words = len(WORD.findall(text))
        fp = first_person_count(text); ev = len(EVALUATION.findall(text)); nouns = len(REVIEW_NOUNS.findall(text))
        col = len(COLLOQUIAL.findall(text)); exp = len(EXPERIENCE.findall(text))
        dom = domain_of(text)
        # passage gate: venue marker inside the kept text, real first person, an experience cue, >=2 evaluations, >=2 review nouns
        if tier == "strict":
            ok = dom is not None and MIN_WORDS <= n_words <= MAX_WORDS and fp >= 1 and exp >= 1 and ev >= 2 and nouns >= 2
        else:
            ok = dom is not None and MIN_WORDS <= n_words <= MAX_WORDS and (fp >= 1 or col >= 1) and (exp >= 1 or ev >= 3) and ev >= 2 and nouns >= 1
        if not ok:
            stats["gate_fail"] += 1; continue
        stats["kept"] += 1
        passages.append(Passage(text=text, domain=dom, n_words=n_words, first_person=fp, evaluation=ev, review_nouns=nouns,
                                colloquial=col, experience=exp, kept_sentences=best[1] - best[0], total_sentences=len(sents)))
    return passages, stats
