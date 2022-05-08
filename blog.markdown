---
# This page will list my blog posts
layout: default
title: Blog
permalink: /blog/
---

# Blog

### Hello! 🐌 
On this page you'll find a listing of all my blog posts.
This blog is my separate standalone version of my Medium blog, which you can find [here](https://medium.com/@jayyydyyy).

In addition to crossposts of all my blog posts, I'll occassionally post some of my running notes and little writings that might note constitute a whole blog post!

---

## Blog Posts:

<ul>
  {% for post in site.posts %}
    {% if post.tag != 'note' %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
    {% endif %}
  {% endfor %}
</ul>

## Notes:

<ul>
  {% for post in site.posts %}
    {% if post.tag == 'note' %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
    {% endif %}
  {% endfor %}
</ul>