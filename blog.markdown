---
# This page will list my blog posts
layout: default
title: Blog
permalink: /blog/
---

# Blog
## Hello! 🐌
### On this page you'll find a listing of my blog posts!
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>