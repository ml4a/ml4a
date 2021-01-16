<h1 align="center">
  <br>
  <a href="https://ml4a.net/"><img src="https://pbs.twimg.com/profile_images/717391151041540096/K3Z09zCg_400x400.jpg" alt="ml4a" width="200"></a>
  <br>
  Machine Learning for Artists
  <br>
</h1>
<div align="center">
    <a href="https://ml-4a.slack.com/"><img src="https://img.shields.io/badge/chat-on%20slack-7A5979.svg" /></a> 
    <a href="https://mybinder.org/v2/gh/ml4a/ml4a/ml4a.net"><img src="https://mybinder.org/badge.svg" /></a> 
    <a href="http://colab.research.google.com/github/ml4a/ml4a/blob/ml4a.net"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
    <a href="https://twitter.com/ml4a_"><img src="https://img.shields.io/twitter/follow/ml4a_?label=Follow&style=social"></a>
</div>

[ml4a](https://ml4a.net) is a Python library for making art with machine learning. It features:

* an API wrapping popular deep learning models with creative applications, including [StyleGAN2](https://github.com/NVLabs/stylegan2/), [SPADE](https://github.com/NVlabs/SPADE), [Neural Style Transfer](https://github.com/genekogan/neural_style), [DeepDream](https://github.com/genekogan/deepdream), and [many others](https://github.com/ml4a/ml4a/tree/master/ml4a/models/submodules).
* a collection of [Jupyter notebooks](https://github.com/ml4a/ml4a-guides/tree/ml4a.net/examples) explaining the basics of deep learning for beginners, and providing [recipes for using the materials creatively](https://github.com/ml4a/ml4a-guides/tree/ml4a.net/examples/models).

## Example

ml4a bundles the source code of various open source repositories as [git submodules](https://github.com/ml4a/ml4a-guides/tree/ml4a.net/ml4a/models/submodules) and contains wrappers to streamline and simplify them. For example, to generate sample images with StyleGAN2:

```
from ml4a import image
from ml4a.models import stylegan2

network_pkl = stylegan2.get_pretrained_model('ffhq')
stylegan2.load_model(network_pkl)

samples, _ = stylegan2.random_sample(3, labels=None, truncation=1.0)
image.display(samples)
```

Every model in `ml4a.models`, including the `stylegan2` module above, imports all of the original repository's code into its namespace, allowing low-level access.

## Support ml4a

### Become a sponsor

You can support ml4a by [donating through GitHub sponsors](https://github.com/sponsors/ml4a/). 

### How to contribute

Start by joining the [Slack](https://join.slack.com/t/ml-4a/shared_invite/enQtNjA4MjgzODk1MjA3LTlhYjQ5NWQ2OTNlODZiMDRjZTFmNDZiYjlmZWYwNGM0YjIxNjE3Yjc0NWVjMmVlZjNmZDhmYTkzZjk0ZTg1ZGM%3E) or following us on [Twitter](https://www.twitter.com/ml4a_). Contribute to the codebase, or help write tutorials.


## License

ml4a itself is [licensed MIT](https://github.com/ml4a/ml4a/blob/master/LICENSE), but you are also bound to the licenses of any [models](https://github.com/ml4a/ml4a/tree/master/ml4a/models/submodules) you use.
