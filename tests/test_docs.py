"""
Test that the code in the docs runs
"""
import re
import matplotlib.pyplot as plt
import pytest
import sciris as sc

root = sc.thispath(__file__).parent
pages = ["README.md", "docs/getting_started.md"]


@pytest.mark.parametrize("page", pages)
def test_docs(page):
    sc.heading(f"Testing the code in {page}")
    text = (root / page).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) > 0  # Each page has code
    namespace = {}  # Shared, since a page's blocks build on each other
    for block in blocks:
        exec(block, namespace)
    plt.close("all")


if __name__ == "__main__":
    T = sc.timer()
    for page in pages:
        test_docs(page)
    T.toc()
