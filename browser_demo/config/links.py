"""
Link token configuration for Browser LM.

The model outputs special tokens like <|link:project:smulgrad|> which get
post-processed into actual hyperlinks. This keeps the model simple while
enabling rich output with clickable links.

URL Structure (assuming chris.dev as domain):
- Projects: chris.dev/projects/{slug}
- Blog: chris.dev/blog/{slug}
- GitHub: github.com/0Chris5R/{repo}
"""

# Base URLs - update these when the website is deployed
BASE_URL = "https://chris.dev"  # Update with actual domain
GITHUB_USERNAME = "0Chris5R"

# Link token format: <|link:{type}:{identifier}|>
# Types: project, blog, github

# Project slug mappings
PROJECTS = {
    "smulgrad": {
        "title": "SmulGrad",
        "slug": "smulgrad",
        "description": "Automatic differentiation from scratch",
    },
    "transformer": {
        "title": "Transformer from Scratch",
        "slug": "transformer",
        "description": "Stanford CS336 Assignment 1 implementation",
    },
    "ml-systems": {
        "title": "ML Systems Optimization",
        "slug": "ml-systems",
        "description": "Stanford CS336 Assignment 2 - DDP, FlashAttention",
    },
    "reflecta": {
        "title": "Reflecta",
        "slug": "reflecta",
        "description": "AI-powered smart journaling app",
    },
    "browser-lm": {
        "title": "Browser LM",
        "slug": "browser-lm",
        "description": "Personal language model for the browser",
    },
}

# GitHub repository mappings
GITHUB_REPOS = {
    "smulgrad": "smulgrad",
    "StanfordCS336-Own-Transformer": "StanfordCS336-Own-Transformer",
    "StanfordCS336-assignment2-systems": "StanfordCS336-assignment2-systems",
    "Reflecta_SmartJournaling": "Reflecta_SmartJournaling",
}

# Blog post mappings (add as needed)
BLOG_POSTS = {
    # "attention-paper": {
    #     "title": "Understanding Attention",
    #     "slug": "attention-explained",
    # },
}


def get_link_url(link_type: str, identifier: str) -> str | None:
    """
    Convert a link token to an actual URL.

    Args:
        link_type: One of 'project', 'blog', 'github'
        identifier: The identifier for the resource

    Returns:
        The full URL, or None if not found
    """
    if link_type == "project":
        if identifier in PROJECTS:
            return f"{BASE_URL}/projects/{PROJECTS[identifier]['slug']}"

    elif link_type == "blog":
        if identifier in BLOG_POSTS:
            return f"{BASE_URL}/blog/{BLOG_POSTS[identifier]['slug']}"

    elif link_type == "github":
        if identifier in GITHUB_REPOS:
            return f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPOS[identifier]}"

    return None


def get_link_html(link_type: str, identifier: str) -> str:
    """
    Convert a link token to an HTML anchor tag.

    Returns the original token if no mapping exists.
    """
    url = get_link_url(link_type, identifier)
    if url is None:
        return f"<|link:{link_type}:{identifier}|>"

    # Get a nice title
    if link_type == "project" and identifier in PROJECTS:
        title = PROJECTS[identifier]["title"]
    elif link_type == "blog" and identifier in BLOG_POSTS:
        title = BLOG_POSTS[identifier]["title"]
    elif link_type == "github":
        title = f"GitHub: {identifier}"
    else:
        title = identifier

    return f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'


def process_link_tokens(text: str, output_format: str = "html") -> str:
    """
    Process all link tokens in text and replace with actual links.

    Args:
        text: Text containing link tokens like <|link:project:smulgrad|>
        output_format: 'html' for anchor tags, 'markdown' for [title](url), 'url' for just URLs

    Returns:
        Text with link tokens replaced
    """
    import re

    # Pattern matches <|link:type:identifier|>
    pattern = r'<\|link:(\w+):([^|]+)\|>'

    def replace_link(match):
        link_type = match.group(1)
        identifier = match.group(2)

        url = get_link_url(link_type, identifier)
        if url is None:
            return match.group(0)  # Return original if no mapping

        # Get title
        if link_type == "project" and identifier in PROJECTS:
            title = PROJECTS[identifier]["title"]
        elif link_type == "blog" and identifier in BLOG_POSTS:
            title = BLOG_POSTS[identifier]["title"]
        elif link_type == "github":
            title = identifier
        else:
            title = identifier

        if output_format == "html":
            return f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
        elif output_format == "markdown":
            return f'[{title}]({url})'
        else:  # url
            return url

    return re.sub(pattern, replace_link, text)


# Special tokens to add to tokenizer
LINK_SPECIAL_TOKENS = [
    # Project links
    "<|link:project:smulgrad|>",
    "<|link:project:transformer|>",
    "<|link:project:ml-systems|>",
    "<|link:project:reflecta|>",
    "<|link:project:browser-lm|>",
    # GitHub links
    "<|link:github:smulgrad|>",
    "<|link:github:StanfordCS336-Own-Transformer|>",
    "<|link:github:StanfordCS336-assignment2-systems|>",
    "<|link:github:Reflecta_SmartJournaling|>",
    # Blog link placeholder (add specific posts as needed)
    "<|link:blog:placeholder|>",
]
