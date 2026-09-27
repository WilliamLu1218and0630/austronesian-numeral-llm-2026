# Prompt 設計

## Zero-shot（不給名、不搜尋）

```
Given the following numeral forms (1–10) of a language, analyze the morphological structure of each numeral and determine its formation pattern.

For each numeral, write its decomposition formula:
- If the numeral is a basic/atomic word (not derived from other numerals), write the number itself (e.g., "1", "5")
- If the numeral is derived from other numerals, write the formula using +, -, × (e.g., "5+1", "2×3", "10-1")

Numerals:
1: {w1}
2: {w2}
3: {w3}
4: {w4}
5: {w5}
6: {w6}
7: {w7}
8: {w8}
9: {w9}
10: {w10}

Reply with ONLY the 10 formulas, comma-separated, in order from 1 to 10.
```

## One-shot（附 Seediq 範例、不給名、不搜尋）

```
Given the following numeral forms (1–10) of a language, analyze the morphological structure of each numeral and determine its formation pattern.

For each numeral, write its decomposition formula:
- If the numeral is a basic/atomic word (not derived from other numerals), write the number itself (e.g., "1", "5")
- If the numeral is derived from other numerals, write the formula using +, -, × (e.g., "5+1", "2×3", "10-1")

Example:
Input: 1: kiŋal, 2: daha, 3: teru, 4: sepac, 5: rima, 6: mumuteru, 7: mupitu, 8: mumusepac, 9: muŋari, 10: maxal
Output: 1, 2, 3, 4, 5, 2×3, 7, 2×4, 9, 10
(6 "mumuteru" contains mu-mu-teru where teru=3, suggesting 2×3; 8 "mumusepac" contains mu-mu-sepac where sepac=4, suggesting 2×4)

Now analyze this language:
1: {w1}
2: {w2}
3: {w3}
4: {w4}
5: {w5}
6: {w6}
7: {w7}
8: {w8}
9: {w9}
10: {w10}

Reply with ONLY the 10 formulas, comma-separated, in order from 1 to 10.
```

## Prompt 變體：neutral（priming ablation，2026-07-14 加入）

動機：8B 煙霧測試大量幻覺 `10-x` 模板，懷疑記號範例（"5+1", "2×3", "10-1" 都是 1-10 的真實候選答案）priming 弱模型。中性版把示範全換成 1-10 範圍外的數字：

```
- If the numeral is a basic/atomic word (not derived from other numerals), write the number itself (for instance, a language whose word for eleven is a single unanalyzable word would get "11")
- If the numeral is derived from other numerals, write the formula using +, -, × (for instance, a word for twelve built from 'ten' plus 'two' would be "10+2"; a word for twenty built from 'two' times 'ten' would be "2×10"; a word for nineteen meaning 'twenty minus one' would be "20-1")
```

用法：`--prompt-variant neutral`，結果檔名帶 `_neutralprompt` 後綴。
設計：小模型＋大模型各跑同一批語言的兩版 prompt，比 FP 率 → 區分「範例誘導」vs「模型天性」。

## 條件變體

### 給語言名稱

在 "Numerals:" 前加一行：
```
Language: {language_name}
```

### 開啟搜尋

在 prompt 開頭加：
```
You may search the web for information about this language if needed.
```
（搜尋功能由各模型 API 的 tool/grounding 設定控制）

## 注意事項

- Prompt 語言統一用英文（跨模型公平性）
- 輸出格式嚴格限制為逗號分隔公式（方便自動解析）
- Seediq 作為 one-shot 範例時，Seediq 本身仍在測試集中（但 AI 已看過正確答案）
  - 論文中需說明此點，或從 one-shot 結果中排除 Seediq
- Temperature = 0 for all models
