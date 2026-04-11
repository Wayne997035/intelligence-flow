import unittest
from src.deliverers.notion_sender import NotionSender

class TestNotionBlocks(unittest.TestCase):
    def setUp(self):
        # 初始化 NotionSender，但不提供 Token 以免真的發送
        self.sender = NotionSender()

    def test_parse_blocks_structure(self):
        """測試解析邏輯生成的 Blocks 結構是否正確"""
        mock_text = """
[SECTION_SUMMARY]
這是市場摘要內容。

[NEWS_ITEM]
TITLE: 測試標題
URL: https://example.com/news
SUMMARY: 這是測試摘要。
INSIGHT: 這是測試洞察。

[EXPERT_VIEW]
這是專家總結。
"""
        # 執行解析 (我們測試私有方法 _parse_and_build_blocks)
        blocks = self.sender._parse_and_build_blocks(mock_text, "測試報告", "blue_background")
        
        # 驗證結構
        # 尋找 Heading 3 (標題)
        h3_blocks = [b for b in blocks if b['type'] == 'heading_3']
        self.assertEqual(len(h3_blocks), 1)
        
        # 驗證標題是否為純文字 (不應包含 link 屬性)
        # 目前的代碼中是：{"text": {"content": title, "link": {"url": url}}}
        # 我們希望改掉它
        rich_text = h3_blocks[0]['heading_3']['rich_text'][0]
        title_content = rich_text['text']['content']
        title_link = rich_text['text'].get('link')
        
        print(f"\n[測試] 標題文字: {title_content}")
        print(f"[測試] 標題連結: {title_link}")
        
        # 尋找 Callout 內的連結
        callout_blocks = [b for b in blocks if b['type'] == 'callout']
        self.assertEqual(len(callout_blocks), 1)
        
        # 驗證 Callout 底部是否有連結
        callout_rich_text = callout_blocks[0]['callout']['rich_text']
        has_link_in_callout = any('link' in rt.get('text', {}) for rt in callout_rich_text)
        print(f"[測試] Callout 內是否有連結: {has_link_in_callout}")

if __name__ == '__main__':
    unittest.main()
