import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.services.url_parser import parse_single_url, parse_website_social_links


def test_url_parser():
    print("=== Testing Task 3: URL Parser, Validation, and Platform Identification ===")

    # 1. One website URL
    res1 = parse_single_url("https://example.com")
    assert res1.is_valid is True
    assert res1.platform == "website"
    assert res1.normalized_url == "https://example.com"
    print("✓ Test 1 Passed: Single website URL")

    # 2. Instagram URL (with www and without)
    res2a = parse_single_url("https://instagram.com/brand")
    assert res2a.is_valid is True
    assert res2a.platform == "instagram"
    assert res2a.normalized_url == "https://instagram.com/brand"

    res2b = parse_single_url("https://www.instagram.com/brand")
    assert res2b.is_valid is True
    assert res2b.platform == "instagram"
    assert res2b.normalized_url == "https://www.instagram.com/brand"
    print("✓ Test 2 Passed: Instagram URL (with/without www)")

    # 3. Facebook URL
    res3 = parse_single_url("https://facebook.com/brand")
    assert res3.is_valid is True
    assert res3.platform == "facebook"
    assert res3.normalized_url == "https://facebook.com/brand"
    print("✓ Test 3 Passed: Facebook URL")

    # 4. LinkedIn URL
    res4 = parse_single_url("https://linkedin.com/company/brand")
    assert res4.is_valid is True
    assert res4.platform == "linkedin"
    assert res4.normalized_url == "https://linkedin.com/company/brand"
    print("✓ Test 4 Passed: LinkedIn URL")

    # 5. Unknown domain (valid HTTP URL on un-categorized platform domain e.g. twitter/custom)
    res5 = parse_single_url("https://mycustomblog.io/about")
    assert res5.is_valid is True
    assert res5.platform == "website"
    assert res5.normalized_url == "https://mycustomblog.io/about"
    print("✓ Test 5 Passed: Unknown/custom domain website URL")

    # 6. Invalid URL (e.g. invalid string format or invalid scheme)
    res6a = parse_single_url("not_a_valid_url")
    assert res6a.is_valid is False
    assert res6a.platform == "unknown"
    assert res6a.normalized_url is None
    assert res6a.error is not None

    res6b = parse_single_url("ftp://example.com/file")
    assert res6b.is_valid is False
    assert res6b.platform == "unknown"
    assert res6b.normalized_url is None
    print("✓ Test 6 Passed: Invalid URL & unsupported scheme handling")

    # 7. Multiple URLs, Empty lines, and Leading/Trailing whitespace
    raw_input = """
      https://example.com  

    https://instagram.com/brand   
    
    https://facebook.com/brand
    
      https://linkedin.com/company/brand  
    invalid-url-line
    """
    parsed_list = parse_website_social_links(raw_input)

    assert len(parsed_list) == 5, f"Expected 5 non-empty items, got {len(parsed_list)}"

    # Item 0: website
    assert parsed_list[0].original_url == "https://example.com"
    assert parsed_list[0].platform == "website"
    assert parsed_list[0].is_valid is True

    # Item 1: instagram
    assert parsed_list[1].original_url == "https://instagram.com/brand"
    assert parsed_list[1].platform == "instagram"
    assert parsed_list[1].is_valid is True

    # Item 2: facebook
    assert parsed_list[2].original_url == "https://facebook.com/brand"
    assert parsed_list[2].platform == "facebook"
    assert parsed_list[2].is_valid is True

    # Item 3: linkedin
    assert parsed_list[3].original_url == "https://linkedin.com/company/brand"
    assert parsed_list[3].platform == "linkedin"
    assert parsed_list[3].is_valid is True

    # Item 4: invalid URL line
    assert parsed_list[4].original_url == "invalid-url-line"
    assert parsed_list[4].platform == "unknown"
    assert parsed_list[4].is_valid is False

    print("✓ Test 7 Passed: Multi-line string with whitespace, empty lines, and mixed URLs")

    print("\n=== All Task 3 URL Parser Unit Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_url_parser()
