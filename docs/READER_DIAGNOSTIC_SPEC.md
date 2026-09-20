# Chẩn đoán C4: bằng chứng mới hay thay đổi điểm vận hành?

## Câu hỏi và phạm vi

Root chốt câu hỏi: C4 multiple full-forward readers có precision/recall envelope tốt hơn mộtCLS, hay gainrecall chỉ là trade-off mà baseline đã đạt bằng threshold? Đây là diagnosis trước đề xuất mới, không huấn luyện C4, không claimnovelty. Giữ model/dataset/E17/independentreview artifacts nguyên vẹn.

Execution agent Luna xhigh thực hiện extraction/computation; root tự đọc kết quả, lựa giả thuyết và research. Reuse agent E17recovery, không spawn helper. GPU ưu tiên sau Hotel nếu chưa rảnh; không dừng jobs khác.

## Inputs

Student E8 seed42 và original BamiBERT E14 selected LR5e-5 trên ba ABSA tasks; paths theo scripts/independent_review/mechanism_battery.py. Validationonly cho lượt extraction này, tuyệt đối khôngtest. Root biết đây là diagnostics trên validation đã xem, không held-out confirmation.

Đọc kỹ script clause_readers.py và MaskedCLS trong fork_test.py. **Không import script có top-level model loops**; tách phần cần dùng vào helper isolated mới, ghi hashes nguồn tham chiếu. Không vô tình chạy lại toàn battery do importsideeffects.

C4 hiện là replicated encoder forwards dùng cùngweights, không shared computedtokenstack. Reproduce fullreader + clause readers của C4 hiện tại, không sửa clause rules/masks/head/decoder. model.eval,outerBF16,manualattentionFP32 nhưreference; đọc đúngcodes0absent1pos2neg3neutral. ViSFD OTHERS index5 binary, excluded pairmetric.

## Extraction

Save per-original-row IDs/gold/textSHA, everyreader clause span/keep-tokenmask và rawslotlogits/predictprobs, fullreaderindex0, sốreader. Cache compressedNPZ/PT plusJSONindex; không mởpublicfile chứaPII tùy tiện. Outputdirectory `artifacts/research_diagnostics/reader_operating_point/` mới.

Baseline vàC4 dùng fullreader path cùngmanualattention, cùngbatchsize/precision đểcomparisonkhớp. Không cần nativeSDPA rerun nếu existing parityoutputs đã đủ; ghi precisioncaveat. Verify knowncounts771/795/1112 vàIDs/gold theo frozenloader, uniqueIDs. Computeδ0metric so với C4reference gầntrong.1pp; nếukhác hơn investigate batching/implementation trước diễn giải, khôngthayexpectedgold.

## Primary policies

1. full: reader0.
2. C4maxpresent: reader r*=argmax_r(1−softmax(logits_r)[absent]) cho từngslot; tie chọnreaderđầu nhưoldcode. Không tự thay bằngmaxpolaritymargin hoặc logprob khi ties; báo ties.

Với reader đãchọn, positivepolarity c*=argmax_{c=1..3} z_c, margin m=maxpositive z−z_absent. Predict c* nếu m>δ, elseabsent. δ0 khớp original argmax decoder (tieabsent). ViOTHERS bỏ khỏiPRaggregation.

## PR diagnosis, không calibration được đưa vào benchmark

Xuất empirical PR/F1 curve theo **globalδ**, group equalmargin để ties xửlýđúng. ScoredgoldcountG fixed; sort allslotsdescendingmargin, predcountP cumulatively, correctpairTP chỉgold>0 vàc*=gold. Pprecision=TP/P,R=TP/G,F1=2TP/(P+G).

- Báoδ0 metrics, allcurve envelope, baselinebestprecision ởrecall≥C4δ0recall vàC4bestprecisionởrecall≥baselineδ0recall.
- Báo maxF1 của mỗi curve nhưng gắn nhãn **oracle validation diagnostic**, không selected benchmark score vàkhôngđemthreshold này sangtest.
- Nativeδ0 minority/multi errors: missed,wrongpolarity,spurious; single/multi goldaspects; mixed/nonmixed polarity; reader-count bins1,2,3–4,≥5 với supportcount. Không diễn giải crossbin nhưcausal.
- So actualchanges từfull→C4: missed corrected, newlyspurious, correct polarity broken, wrongpolarityfixed, samepred. Save perrow/peraspect table hỗtrợrootmanualreview.
- Báo concentration: mộtreader có support/goldaspect nào, nhưng **không gán sentencepositive label cho mọiclause**.

Không quét hyperparameter model, không fitclassifier hoặctrainencoder, không choose finalthreshold từval. Đây là cách kiểm lời giải thích “chỉvì dễ dựđoánpresent hơn”.

## Deliverables và vận hành

`scripts/research_checks/reader_operating_point.py`; percheckpoint caches/provenance,curvesCSV, error-transitionJSON/CSV,summaryJSON vàfactualREPORT.md. Modelsourceweights SHA trước/sau, scriptSHA, actualGPUtime/peakmemory; no qualityinterpretation/noveltyclaim từagent.

Nếu extraction dài, dùng detachedlauncher stdinDEVNULL/start_new_session=True, ghiactualPID/command/log; incremental cache theo model/task đểresume đúngcompletedscopes. Khôngrerun cachekhớp provenance. Gửi rootkếtquả từngtask ngay khi có; không chờ đủ6đểbáo lỗi. E17recovery không bịgiánđoạn.
