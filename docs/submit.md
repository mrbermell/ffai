# Submit Your Bot to Bot Bowl III

## Bots as Servers

You must use the [submission template](https://github.com/njustesen/bot-bowl-submission) and 
modify the run.py script so that it imports your own bot. Simply replace the line ```import mybot``` with ```import <your-bot-script>```.

In your bot script, make sure your bot is registered with a suitable name for your bot:
```python
# Register the bot to the framework
ffai.register_bot('my-random-bot', MyRandomBot)
```

In Bot Bowl II, every bot plays a number of games against each other bot. During each series of ten games, the two competing bots will 
be instantiated just once, play the ten games of the series, and finally shut down. Before each new game in the series, the ```new_game()``` method is called and at the end of each game, ```end_game()``` is called. 
This allows you to easily adapt your strategy based on the game result without having to save and load files. 

If you want to understand how the competition server works in more detail, take a look at the [competition example](https://github.com/njustesen/ffai/blob/master/examples/competition_example.py).

## Submission link

Will appear some time in June.

## Verification
Within one week after the submission deadline, we will test your bot against the  baseline random and contact you to verify that the results are as expected. 
In case we did not setup your bot in the right way, please set aside some time to assist us.
