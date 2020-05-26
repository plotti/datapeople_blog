---
Title: Chatbots - how to build them. The stupid, the smart and the ugly.
Description: The chatbot hype is gone. Or is it? What left and what makes sense? Get an overview of the opensource and Saas chatbot landscape and follow a complete tutorial on how to build your own chatbot with chatfuel and a backend architecture.
Date: 11 January 2019
Author: Thomas Ebermann
Img: bot.jpg
Template: post
Tags: data,blog
---
Chatbots have been quite the trend since 2017. Several [swiss](http://www.inside-it.ch/articles/49513) [companies]( http://www.inside-it.ch/articles/49015) have launched their chatbot services, partly with mixed experiences. [Experts](https://www.grandviewresearch.com/press-release/global-chatbot-market) expect that the global chatbot market is growing from 50 Million dollars (2015) up to 1.5 Billion dollars in 2025. The main reason  is of course the reduction of customer service costs. The rationale is simple, if the chatbot can handle customer requests for free, you will need less staff. 

Although its 2019 - and the chatbot trend might seemed have cooled off a bit - I thought I'd still cover some of my experiences of building a chatbot for Facebook. Its purpose is suggesting a game you can play with kids.  It asks you in which context you are in right now (e.g. Outdoors with 2 Kids / In the car with your kid / Waiting room etc..) and then the chatbot suggests you a game you might play together. 

But before diving into the project, this article will provide you with a bit of background information on chatbots and explains why it is hard for machines to understand use first. It will then provide an overview of different options if you want to build a chatbot: Go for an open source solution or use one of the many services available. We will finally cover an implementation of a simple chatbot with chatfuel and finish with an evaluation and lessons learned before providing an outlook.

## Overview

If you  follow the “chatbot” topic closely, you might have noticed, that companies who have invested in this technology went through the typical [Gartner hype cycle](https://www.gartner.com/technology/research/methodologies/hype-cycle.jsp). In 2018 you can expect most of them to follow through the valley of [disillusionment]( https://www.finews.ch/news/banken/30995-chatbots-cs-ubs-swissquote-cler-postfinance-amelia-carl-rose). So what happened - you might ask? Well it turns out in the majority of cases chatbots simply couldn't live up the hype. After all, building a chatbot which works as well as a personal assistant on  the phone, turned out to be as hards as making a machine pass the [Turing test](https://de.wikipedia.org/wiki/Turing-Test). 

### Why is it still hard for machines to understand us?

So why is it so hard to build a chatbot that understands us? After all we have seen huge improvements in language oriented machine learning tasks. For example Google was able to launch a new translation product whose product lead (Barak Turovsky) is described as: "It has improved more in one single leap than in 10 years combined". This big leap forward in translation in 2016, was mainly based on the use of neural networks that allowed to consolidate translation efforts in one big [system](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/): Here Google started to use deep learning systems (in particular [Long Short Term Memory networks]( http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (LSTMS), which are a special form of recurrent neural networks (RNNs)) and achieved outstanding results, as shown below:

![Translation](%assets_url%/translation.jpg)

I suspect that a lot of people saw this leap forward as a sign that the computer finally understands us.  Unfortunately this is still not the case yet. While neural networks have learned to rather translate sentence by sentence and not word by word and map whole sentences from one language to another their task was never, to actually understand what we mean by our words. So right now we are at the point where we can use RNNs to [mimick]( http://karpathy.github.io/2015/05/21/rnn-effectiveness/) human language extremely well, but still not really understanding it. 

All these progresses- although groundbreaking - unfortunately mean that chatbots have been hyped for “AI” in 2017 - still have a long way to go before we can have a meaningful conversations with them. Until then, they are to a certain extent able to distinguish the [named entities]( https://en.wikipedia.org/wiki/Named-entity_recognition) (e.g. Barack Obama, Zürich, Flight to Frankfurt) that are present in a sentence and finding out the intent (basically what you want to do with that entity). In this task they are rather still limited to cookie cutter phrases like "What time is it in Beijing?". While [natural language understanding](https://en.wikipedia.org/wiki/Natural_language_understanding) will advance rapidly for sure, for chatbots it turns out to be a good idea to bear two things in mind:

- limit the chatbot application to a very narrow context (e.g. in our example selecting games to play)
- make use of "canned" quick replies to narrow down the possibilities of misunderstanding a user and save him some typing.

That is why in this blog post I have chosen to build one of these "dumb" chatbots which is  using a very narrow context and make heavy use of quick replies to get the user to navigate through a big solution space (in our case a database of games).

## Open Source Solutions

By trying something new, as a developer you want to give the open source solutions a try first. In this case you are particularly lucky, since there are quite a few solutions available. Let me show you three: 

1.There are options like [P-Brain]( https://github.com/patrickjquinn/P-Brain.ai), that uses node and bootstrap and roll the chatbot logic and the visual representation into one handy framework that you can use in your browser. 
2.[Botpress](https://botpress.io) offers you a very developer friendly and modular (e.g. natural language understanding, analytics, etc..) node solution (think of rake for chat bots :)
3. [Botkit]( https://botkit.ai) offers you a full stack solution with a dialog editor, analytics and various plugins - all rolled into one node module. 

The good thing about these open source frameworks is that as a developer or business owner you are in control: You control the data and the logic and you are flexible on where to integrate the solution and might even save some money. A big benefit is also that all of those solutions can easily be connected to a number of Platforms such as [Slack]( http://slack.com), [Jabber](https://www.jabber.org), [Twilio]( https://www.twilio.com) , [Facebook Messenger](http://messenger.com) or generic web apps of course. The latest being actually quite a promising way of reducing complexity on websites (thinking of those big menu bars, with millions of sub-menus) or offering a smart and conversational search option. 

## As a service Solutions

If you have been living under a rock, you might expect some cool startups in the chatbots as a service area. But the reality is that we are probably 2-3 years to late for this game. So as usual the big 5 (Facebook, Amazon, Microsoft, Google and IBM) have bought the most promising candidates on the market and are now competing for your attention. As a developer that might not be the worst thing, since you have a number of options to choose from. So here they are:

1. Facebook offers [wit.ai]( http://wit.ai) as a chatbot framework. Wit.ai allows you to model entities and intent, and offers integration for automatic speech recognition too. 
2. Amazon offers [lex]( https://aws.amazon.com/de/lex/) which is used internally to power Alexa. It also offers automatic speech recognition (ASR) and natural language understanding (NLU). When scaling your service to replace dozens of call centers, the integrated Amazon infrastructure probably makes sense for you :). One cute fact is that they offer also speech synthesis under the cute name "polly". 
3. Microsoft offers two solutions: One of them is [luis.ai]( http://luis.ai), a NLU solution that mainly does entity recognition and intent disambiguation. The other one is the [botframework]( https://dev.botframework.com) which is a framework that integrates with the Microsoft universe (Cortana, Skype, etc.. you see where this is going :))
4. Google offers [api.ai]( http://api.ai) a startup they bought two years ago and rebranded it to dialog flow. It runs on Google's tensorflow infrastructure. 
5. Finally IBM uses their familiar Watson brand to also offer a [chatbot]( https://www.ibm.com/watson/de-de/wie-man-einen-chatbot-entwickelt/) solution, that runs in the IBM cloud. 
6. Even SAP has bought a startup called [racast.ai](https://recast.ai) in this field, which also offers quite a few existing libraries for developers (e.g. go, erlang, ruby, python, etc..).

After reading all of these options and maybe after browsing through the fancy feature descriptions, you might be intimidated thinking: Chatbots are expensive, or that sounds really complicated. Luckily you don't have to be. It's fairly easy to get started. For this blog post I have picked none of the options above, but went with [chatfuel](https://chatfuel.com), because it makes starting so much fun. 

## Lets get started
 
Chatfuel is a chatbot solution that work best for Facebook with their messenger service. Yes it's mainly for Facebook, but you can also integrate it on your website, if you feel like it, and after all you can potentially reach over 1.5 billion people on this platform. In simple cases you don't even need to be able to program to create an easy chatbot. All you need is to login for Facebook to  authenticate chatfuel. Then the dashboard shows up and you can create your first chatbot. 

![New bot](%assets_url%/new_bot.jpg)

Generally among things like; being able to analyze your chatbots performance, trigger promotions, broadcast messages to all users or setup additional attributes for your users,
; the most important part is the conversation flow modeling. That's the part where you connect pre-written text blocks with user inputs and decide which text blocks should follow, based on the response. 

## Conversation Modeling

Usually you start with a welcome message, greeting the user and then sending him to a "block". A block is an area filled with different additional questions which the user has to answer. When I say "has" to answer you already see where this is going. Usually users follow a predefined path through a conversation with having only a few options to "explore" a conversational space. This is possible though, because you can always define fuzzy trigger words or whole sentences such as "I am done here.", that instantly take you to a certain part in the conversation - in this case the end.

Generally it makes sense to draft the conversation to have a general idea how to structure it. It might seem superfluous at first but you might get lost quickly otherwise. So in our case there are basically two questions that we want to ask the user: 

1. Where do they want to play games? Potential answers: "Doesn't matter", "Inside", "Outside", "Waiting Room", "Car", "Kids birthday".
2. Which kind of games do they want to play? Potential answers: "Solo-game", "cooperative" or "competition"

Once the chatbot collected these answers, he can start suggesting games to the user. 

![Flow](%assets_url%/flow.jpg)

As you can see in the image above, the user has been asked where he wants to play games. Based on his response, which is captured by so-called "quick replies", he is sent to another part of the conversation. Both responses are saved to user attributes that we can use to send to our server to trigger a query on our database of games now.  

So you might notice,that the combination of "Kids birthday" games with the type "Solo-Game", does not make much sense  for example unless it's a very lonesome birthday :). You can imagine, that in a more complex scenario, you might encounter more of such cases, where you have to take care of the conversation flow manually. 

## Sinatra Service

Encoding your "data" in the chatbot service is a bad idea (I tried it in the beginning). Maintaining, updating and migrating your data will just be a horrible nightmare. That’s why it makes much more sense, to store it in a database and run a small web-service on top of it, to be able to use it in your chatbot. In our case, I've chosen to use a small [sinatra](http://sinatrarb.com) micro-framework, that offers two routes. One is a /search route that the chatbot uses to query our database of games. The second one is a /show route that allows us to extract games from the database in a way that can be processed by the chatbot. 

### Search

The good thing about chatfuel is, that you can interact with it via a [JSON API]( http://docs.chatfuel.com/plugins/plugin-documentation/json-api). When the user has answered both questions we are sending his response to the server via a simple GET request. 

![Flow](%assets_url%/json.jpg)

On the server we simply query the database for games matching the criteria and respond with a JSON object that chatfuel is able to understand. In this case, we are building a gallery object out of 10 random matching games, that the user can flip through. Once he selects one he will be redirected to the /show route.

```ruby

get "/search" do 
    games = Game.where(:type => params["game_type"]).where(:context => params["context"]).sample(10)
    games.each do |game|
        output << {"title": game["Name"], 
                    "image_url": "http://xxxxx.herokuapp.com/images/resized/" + game["Bild"],
                    "subtitle": game["Kurzbeschreibung"],
                    "buttons": [{"type": "json_plugin_url","url": "http://xxxxx.herokuapp.com/show?id=#{game["ID"]}","title": "Das will ich spielen!"}]
                  }
    end
    content_type :json
    {
    messages: [
      {
      "attachment":{
        "type":"template",
        "payload":{
          "template_type":"generic",
          "image_aspect_ratio": "square",
          "elements": output
        }
       }
     }
    ]}.to_json
    end
```
### Show 

Once the user has selected a game, we are showing  a matching image and the game description to him. Using the JSON API we can very conveniently also supply more messages to the user, in this case the simple question is, if he wants another game or wants to end his conversation. Interesting fact to notice is, that we can point the user to a "block" in chatfuel directly, where the conversation is picked up. In our case the block "configure" starts by asking the user again (the same two questions), allowing to find other games.

```ruby
get "/show" do 
    game = Game.where(:id => params["id"]).first
    content_type :json
    {
      messages: [
        {
         text: game["Kurzbeschreibung"]
        },
        {
          attachment: {
            type: "image",
            payload: {
              url: "http://xxx.herokuapp.com/images/" + game["Bild"]
            }
          }
        },
        {
            "attachment": {
              "type": "template",
              "payload": {
                "template_type": "button",
                "text": "Willst Du noch eins?",
                "buttons": [
                  {
                    "type": "show_block",
                    "block_name": "configure",
                    "title": "Ja."
                  },
                  {
                    "type": "show_block",
                    "block_name": "danke",
                    "title": "Nein Danke."
                  }
                ]
              }
            }
       }
      ]
    }.to_json
end
```

## Integration into Facebook

Of course after deploying your code to the server (in my case I've simply used heroku) you are ready to go. The good thing about chat fuel is, that all changes to the conversation flow are saved quickly and you can directly test your bot in [Facebook Messenger]( http://messenger.com). From a testing perspective its rather tedious to write automated tests, that guarantee the functionality of your chatbot, since this might means to go down every conversational path and making sure that the responses make sense. Once you are happy with the results, you can interact directly with your bot, by addressing him with his name (that you have defined in chatfuel) in messenger. From here, you might as well create a Facebook page or community and integrate the communication with your chatbot prominently in the header of the page.

![Facebook](%assets_url%/facebook.jpg)

## Evaluation and Conclusion

Chatfuel and Facebook are offering quite a bouquet of analytics for your chatbot application. Beyond simple metrics, like the total number of users, the number of new users and general user activity, you can also compute user retention. Generally you want users to come back and interact with your bot. If they are not coming back,  something might be wrong with the user experience. To investigate such problems chatfuel offers a way to see user inputs that have not been recognized by the system or shows which buttons or blocks have been used the most. 

Similar to a website, where a high percentage of users might e.g. not scroll below a certain point, in a chatbot scenario users might stop interacting with the chatbot once they feel that the conversation is not getting them closer to their goal.  Having to answer more than 3 questions before receiving your answer/reward etc. is very tiring for the end-user, at least that’s what I experienced. Repetitively typing responses, or having to think of responses that are understood by the bot, might become a burden for the user too. In general, I think providing a smooth user experience is a very thin and tricky line to walk. I've only seen a few examples where this is done right. I personally think,  a bot can shine in scenarios especially, where you want users to navigate through a vast and complex solution space, while not overwhelming them with the variety of options. After all, chatting with a chatbot, should not feel like filling out a long form.

## Whats next?

Trying to predict the success of chatbots in different scenarios, I think they will be made use of in customer service heavily. They might either be used to dispatch the customers as a way of 0-level support or even be able to answer some very easy and popular problems right at the start of a conversation maybe to reduce the workload of the staff. A few solutions out there are able to seamlessly handover the conversation to a human already, after they run out of solutions, which might be an interesting hybrid. Yet, personally I feel very annoyed by having to go through these systems to only talk to a human. After all it is hard to replace the touch of humanity with a chatbot system. It is also a question of the value that a customer is having  for a company - e.g. having to talk to a chatbot when wanting to buy an expensive insurance just feels wrong. 

In  different  fields I see bigger potential, especially since social acceptance of those  systems is on the rise, with systems like Cortana, Siri or Alexa. We can probably expect chatbots to turn into full fledged assistants in the commerce area, but we have been waiting since the early 90ies. So although the hype of chatbots is over, we can still expect very interesting and promising scenarios, especially in more mundane commonplace areas, beyond baking,commerce and insurances. So feel free to check out our small [games chatbot](http://facebook.com/gutespielideen/) from this tutorial on Facebook.