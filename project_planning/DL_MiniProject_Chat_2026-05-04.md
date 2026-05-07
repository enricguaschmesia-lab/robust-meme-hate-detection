# WhatsApp Conversation Log
## Chat: Deep learning DL Mini Project
### Date: Yesterday (May 4, 2026)
### Participants: Henrik (+43 650 7158870), Jane EPFL, Enric Guasch (You)

---

**[17:01] Henrik:** guys one question *(edited)*

**[17:02] Henrik:** in the meta dataset the photos are all actual memes where the text is in the image. in my search for datasets, can i also consider those where its a photo with caption, but no text in the image?

**[17:05] Jane EPFL:** like the text is part of the image file but it's not ontop of the photo - it's like a separate chunk?

**[17:05] Henrik:** no

**[17:05] Jane EPFL:** wait nvm

**[17:05] Henrik:** just a normal photo

**[17:05] Henrik:** with a caption

**[17:05] Jane EPFL:** i get what u mean

**[17:05] Jane EPFL:** i think that's ok

**[17:05] Jane EPFL:** it still serves our purpose of making online safer

**[17:06] Enric Guasch:** *(replying to Henrik: "in the meta dataset the photos are all actual memes where the text is in the image. in my search for datasets, can i also consider those where its a photo with caption, but no text in the image?")* Yes I think so, because the way we treat it we only process text in one encoder and text in another and then fusion

**[17:06] Enric Guasch:** So it should be the same

**[17:06] Henrik:** yeah from an ml perspective its the same

**[17:06] Jane EPFL:** but wait why do we want more datasets 😄

**[17:06] Henrik:** im just worrying that the algo learns to read the text on the image yk

**[17:06] Henrik:** *(replying to Jane EPFL: "but wait why do we want more datasets")* because 10k not enough

**[17:07] Henrik:** *(replying to Henrik: "im just worrying that the algo learns to read the text on the image yk")* and then it wouldnt work if theres no text on the image

**[17:08] Henrik:** its fine ill just look

**[17:08] Henrik:** and send what i find

**[17:23] Henrik:** *(forwarded: MAMI@SemEval 2022: Dataset Request — "Your request for the MAMI dataset has been accepted. DOWNLOAD the dataset: [Google Drive link]. PASSWORD: *MaMiSemEval2022! — Best regards, MAMI Organizing Committee")*

ill just leave this here

**[18:57] Henrik:** okay so little summary, I have found quite a bit of data that we can use, and also two models that try to do hate classification for memes. We can try and see if we can use them as a baseline model where we afterwards just train the adversarial one on top.

Datasets:
MultiOFF: [link]
MAMI: [link]
HarMeme: [link] & [link]

Models:
Mememind: [link]
Momenta: [link]

@Enric Guasch maybe you can take a look at the models

**[19:11] Jane EPFL:** I'm more down for Momenta. Mememind is just a dataset but they have their model Memeguard, however it's not CLIP-based. Momenta is kinda CLIP-based but I think there are more recent CLIP models

**[19:11] Jane EPFL:** and I really think we should use CLIP because they talked about it during the multimodal model lecture

**[19:11] Jane EPFL:** and we need to explicitly mention in the report how our project relates to lecture material

**[19:11] Henrik:** great

**[19:12] Jane EPFL:** the Mememind dataset might be good

**[19:12] Jane EPFL:** other than the fact that it's half chinese, but it's catered exactly for our project task

**[19:13] Henrik:** It's not public

**[19:14] Henrik:** But it's a combination of all the ones I've sent

**[19:14] Henrik:** Minus the Chinese part ofc

**[19:15] Jane EPFL:** okok sounds good

**[19:16] Jane EPFL:** should we cite mememind then?

**[19:20] Henrik:** No

**[19:20] Henrik:** Not for the dataset at least

**[19:23] Jane EPFL:** *(shared GitHub link: "gokulkarthik/hateclipper — Hate-CLIPper: Multimodal Hateful Meme Classification with Explicit Cross-modal Interaction of CLIP features - Accepted at EMNLP 2022 Workshop")*

CLIP-based models (state of the art-ish):

HateCLIP: https://github.com/gokulkarthik/hateclipper

MemeCLIP: https://github.com/SiddhantBikram/MemeCLIP

ISSUES: https://github.com/miccunifi/ISSUES

**[22:20] Henrik:** fantastic 😊

---
*Exported on May 5, 2026*
