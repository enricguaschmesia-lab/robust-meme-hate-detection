**Henrik’s Idea: Adversarial Attacks on Multimodal Meme Detection**

idea

* Study how AI models detect offensive memes
* Focus on multimodal models (text + image together)
* Key challenge:
  + A meme might be harmless in text or image alone
  + But offensive when combined
  + How can we trick the model into thinking that offensive models are not offensive 🡪 **Adversarial Attacks**
    - Use common tactics such as censor the input, add noise to the image (as done in class) etc.
    - Text attacks:
      * “hate” → “h@te”
      * “idiot” → “1d!ot”
      * Adding spaces or weird characters
    - Image attacks:
      * Small noise
      * Slight blur
      * Brightness / contrast changes
      * Tiny shifts in text placement
  + Study how text / image alone affect the model, what the reason is for the models‘ output
  + Keep an eye out on the fact that the model doesn’t just say everything is offensive
  + Analyze how performance changes from normal inputs to perturbated ones
* How
  + A multimodal classifier (e.g., CLIP or similar)
  + Input: meme (image + text)
  + Output: offensive vs non-offensive

Datasets to use

* From Enrics list
  + Hateful Memes Dataset
  + MMHS150K
  + HarMeme
  + MAMI (Multimedia Automatic Misogyny Identification)
  + MultiOFF
* Maybe we can even find some others tailored to the task
