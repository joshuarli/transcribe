from transcribe.denoise import _PHONE_RE, _collapse_spoken_digits, normalize_numbers, strip_fillers

# --- single-word disfluencies ---


def test_strips_uh_mid_sentence():
    assert strip_fillers("be they technical uh be they not technical.") == "be they technical be they not technical."


def test_strips_uh_at_start():
    assert strip_fillers("Uh that is the point.") == "That is the point."


def test_strips_um():
    assert strip_fillers("wanting to know about um curing a large cut.") == "wanting to know about curing a large cut."


def test_strips_ah():
    assert strip_fillers("Ah that is interesting.") == "That is interesting."


def test_strips_mm():
    assert strip_fillers("I think, mm, it depends.") == "It depends."


def test_strips_hmm():
    assert strip_fillers("it is, hmm, complicated.") == "it is, complicated."


def test_strips_er():
    assert strip_fillers("I think er you know.") == ""


def test_strips_eh():
    assert strip_fillers("eh I would say that is fine.") == "I would say that is fine."


def test_preserves_uh_huh():
    assert strip_fillers("uh-huh yes I agree.") == "uh-huh yes I agree."


def test_preserves_mm_hmm():
    assert strip_fillers("mm-hmm exactly.") == "mm-hmm exactly."


def test_preserves_words_containing_er():
    assert strip_fillers("the earth is round, ergo it spins.") == "the earth is round, ergo it spins."


def test_strips_multiple_in_sequence():
    assert (
        strip_fillers("and uh we we actually you know have some questions.") == "and we actually have some questions."
    )


def test_cleans_trailing_filler_before_punctuation():
    assert strip_fillers("I think er you know hmm.") == ""


# --- right? tag ---


def test_strips_right_tag_mid_sentence():
    assert strip_fillers("it's old, right? I mean that's the thing.") == "it's old. That's the thing."


def test_strips_right_tag_capitalizes_continuation():
    assert strip_fillers("cut the meat out, right? And then sliced it.") == "cut the meat out. And then sliced it."


def test_preserves_genuine_right_question():
    assert strip_fillers("Is that right? I think so.") == "Is that right? I think so."


# --- you know ---


def test_strips_you_know_mid_sentence():
    assert strip_fillers("I you know think it's good.") == "It's good."


def test_strips_you_know_with_commas():
    assert strip_fillers("I, you know, think so.") == "I think so."


def test_strips_you_know_at_start():
    assert strip_fillers("you know, it takes time.") == "it takes time."


def test_preserves_do_you_know():
    assert strip_fillers("do you know what temp?") == "do you know what temp?"


def test_preserves_do_you_know_capitalized():
    assert strip_fillers("Do You Know what I mean?") == "Do You Know what I mean?"


def test_preserves_if_you_know():
    assert strip_fillers("if you know the answer, share it.") == "if you know the answer, share it."


def test_preserves_if_you_know_capitalized():
    assert strip_fillers("If you know, speak up.") == "If you know, speak up."


# --- stutter dedup ---


def test_dedup_two_char_word():
    assert strip_fillers("we we actually have.") == "we actually have."


def test_dedup_one_char_word():
    assert strip_fillers("I I tend not to like it.") == "I tend not to like it."


def test_dedup_three_char_word():
    assert strip_fillers("the the question came from Caleb.") == "the question came from Caleb."


def test_dedup_four_char_word():
    assert strip_fillers("very very good.") == "very good."


def test_dedup_run_of_three():
    assert strip_fillers("a a a vacuum bag.") == "a vacuum bag."


def test_dedup_much():
    assert strip_fillers("much much quicker penetration.") == "much quicker penetration."


def test_dedup_that():
    assert strip_fillers("that that is fine.") == "that is fine."


def test_dedup_contraction_well():
    assert strip_fillers("we'll we'll get to that.") == "we'll get to that."


def test_dedup_contraction_hes():
    assert strip_fillers("he's he's doing fine.") == "he's doing fine."


def test_dedup_long_word():
    assert strip_fillers("actually actually that's wrong.") == "actually that's wrong."


# --- sentence-level filtering ---


def test_removes_filler_sentence_okay():
    assert strip_fillers("It gets hot. Okay. That makes sense.") == "It gets hot. That makes sense."


def test_removes_filler_sentence_right():
    assert strip_fillers("The crust forms. Right. And then you flip it.") == "The crust forms. And then you flip it."


def test_removes_filler_sentence_yeah():
    assert strip_fillers("Yeah. So the technique is correct.") == "The technique is correct."


def test_removes_call_in_sentence():
    assert strip_fillers("Great technique. Call in with your questions.") == "Great technique."


def test_removes_welcome_to_intro():
    assert (
        strip_fillers("Welcome to Cooking Issues. Today we're talking about salt.") == "Today we're talking about salt."
    )


def test_removes_welcome_back_to_intro():
    assert strip_fillers("Great. Welcome back to the show. Let's continue.") == "Let's continue."


def test_removes_host_of_intro():
    assert strip_fillers("I'm Josh Spiegel, host of the show. Today we discuss salt.") == "Today we discuss salt."


def test_removes_hosted_by_intro():
    assert strip_fillers("Hosted by Dave Arnold. Let's get started.") == "Let's get started."


def test_removes_heritage_radio_sentence():
    assert strip_fillers("Dave Arnold with cooking issues on the Heritage Radio Network. Let's talk.") == "Let's talk."


def test_removes_coming_to_you_live():
    assert strip_fillers("Coming to you live today. So the technique is this.") == "The technique is this."


# --- inline excision for long run-on sentences ---


def test_inline_excise_preserves_content_around_heritage_radio():
    long = (
        "so we were talking about the sous vide technique and on the Heritage Radio Network "
        "the way you set up the water bath is very important for even cooking"
    )
    result = strip_fillers(long)
    assert "sous vide" in result
    assert "water bath" in result
    assert "Heritage Radio" not in result


def test_short_sentence_with_noise_still_dropped():
    assert strip_fillers("Dave Arnold on the Heritage Radio Network.") == ""


def test_inline_excise_call_in_long_sentence():
    long = (
        "you can reach us if you want to call in with a question about fermentation "
        "or pickling and we will answer as many as we can during the show today"
    )
    result = strip_fillers(long)
    assert "fermentation" in result
    assert "pickling" in result
    assert "call in" not in result


# --- spoken digit collapse ---


def test_collapse_ten_digit_phone():
    assert _collapse_spoken_digits("seven one eight four nine seven two one two eight") == "7184972128"


def test_collapse_three_digits():
    assert _collapse_spoken_digits("seven one eight") == "718"


def test_collapse_two_digits():
    assert _collapse_spoken_digits("one two") == "12"


def test_collapse_does_not_touch_single_digit_word():
    assert _collapse_spoken_digits("one cup") == "one cup"


def test_collapse_does_not_touch_compound_number():
    # "sixty" is not a single digit word, so no collapse
    assert _collapse_spoken_digits("sixty three") == "sixty three"


def test_collapse_mixed_format_phone():
    # "seven one eight" → "718", leaving "718-4972128" which _PHONE_RE can catch
    assert _collapse_spoken_digits("seven one eight-4972128") == "718-4972128"


# --- normalize_numbers ---


def test_normalize_two_word_cardinal():
    assert normalize_numbers("sixty three degrees") == "63 degrees"


def test_normalize_compound_hundred():
    assert normalize_numbers("two hundred twenty five degrees") == "225 degrees"


def test_normalize_single_digit():
    assert normalize_numbers("three hours") == "3 hours"


def test_normalize_single_number_word():
    assert normalize_numbers("cook for one hour") == "cook for 1 hour"


def test_normalize_multiple_spans():
    assert normalize_numbers("forty five minutes at sixty three degrees") == "45 minutes at 63 degrees"


def test_normalize_skips_standalone_multiplier():
    assert normalize_numbers("hundred times better") == "hundred times better"


def test_normalize_preserves_trailing_punctuation():
    assert normalize_numbers("cook for forty five.") == "cook for 45."


def test_normalize_skips_non_numbers():
    assert normalize_numbers("the quick brown fox") == "the quick brown fox"


# --- basically / essentially ---


def test_strips_basically_at_start():
    assert strip_fillers("Basically, you want to cook it.") == "You want to cook it."


def test_strips_essentially_at_start():
    assert strip_fillers("Essentially, use less salt.") == "Use less salt."


def test_strips_basically_parenthetical():
    assert strip_fillers("It is, basically, the same thing.") == "It is the same thing."


def test_preserves_basically_without_commas():
    assert strip_fillers("It is basically correct.") == "It is basically correct."


# --- I mean ---


def test_strips_i_mean_at_start():
    assert strip_fillers("I mean, that's the thing.") == "That's the thing."


def test_strips_i_mean_parenthetical():
    assert strip_fillers("It's not that, I mean, it's the opposite.") == "It's not that it's the opposite."


def test_preserves_i_mean_without_trailing_comma():
    assert strip_fillers("I mean it when I say this.") == "I mean it when I say this."


# --- by the way ---


def test_strips_by_the_way():
    assert strip_fillers("It is, by the way, quite good.") == "It is quite good."


# --- or something ---


def test_strips_or_something_at_end():
    assert strip_fillers("I'll use salt or something.") == "I'll use salt."


def test_strips_or_something_like_that():
    assert strip_fillers("Use gelatin or something like that.") == "Use gelatin."


def test_preserves_or_something_mid_clause():
    assert strip_fillers("or something about that matters.") == "or something about that matters."


# --- you know what I mean tag ---


def test_strips_you_know_what_i_mean_tag():
    assert strip_fillers("It's the same thing, you know what I mean?") == "It's the same thing."


def test_strips_what_i_mean_tag():
    assert strip_fillers("It's the same thing, what I mean?") == "It's the same thing."


def test_removes_standalone_you_know_what_i_mean():
    assert (
        strip_fillers("The crust forms. You know what I mean? And then you flip it.")
        == "The crust forms. And then you flip it."
    )


# --- extended filler sentences ---


def test_removes_etc():
    assert strip_fillers("Add some spice. Etc. And stir well.") == "Add some spice. And stir well."


def test_removes_and_so_on():
    assert strip_fillers("You need salt, pepper. And so on. Then cook it.") == "You need salt, pepper. Then cook it."


def test_removes_blah_blah():
    assert strip_fillers("Great technique. Blah blah. Now cook.") == "Great technique. Cook."


# --- connector fragment sentences ---


def test_removes_and_so():
    assert strip_fillers("The meat is ready. And so. We continue.") == "The meat is ready. We continue."


def test_removes_but_then():
    assert strip_fillers("It cooked well. But then. We added salt.") == "It cooked well. We added salt."


def test_removes_go_to_break():
    assert (
        strip_fillers("That's all for now. Go to break. We'll be right back.")
        == "That's all for now. We'll be right back."
    )


# --- fractions ---


def test_normalize_fraction_half():
    assert strip_fillers("use one and a half cups") == "use 1½ cups"


def test_normalize_fraction_quarter():
    assert strip_fillers("cook for two and a quarter hours") == "cook for 2¼ hours"


def test_normalize_fraction_three_quarters():
    assert strip_fillers("add three and three quarters teaspoons") == "add 3¾ tsp"


def test_normalize_fraction_third():
    assert strip_fillers("fill one and a third of the pot") == "fill 1⅓ of the pot"


# --- temperatures ---


def test_normalize_temp_fahrenheit():
    assert strip_fillers("heat to three hundred fifty degrees Fahrenheit") == "heat to 350°F"


def test_normalize_temp_celsius():
    assert strip_fillers("sixty degrees Celsius") == "60°C"


def test_normalize_temp_centigrade():
    assert strip_fillers("sixty degrees Centigrade") == "60°C"


def test_normalize_temp_bare():
    assert strip_fillers("at three hundred fifty degrees") == "at 350°"


def test_normalize_temp_f_abbreviation():
    assert strip_fillers("bake at three hundred fifty degrees F") == "bake at 350°F"


# --- percentages ---


def test_normalize_percent():
    assert strip_fillers("fifty percent salt reduction") == "50% salt reduction"


def test_normalize_percent_compound():
    assert strip_fillers("about twenty five percent") == "about 25%"


# --- measurements ---


def test_normalize_tablespoon():
    assert strip_fillers("add two tablespoons of butter") == "add 2 tbsp of butter"


def test_normalize_teaspoon_plural():
    assert strip_fillers("use three teaspoons of salt") == "use 3 tsp of salt"


def test_normalize_ounce():
    assert strip_fillers("four ounces of flour") == "4 oz of flour"


def test_normalize_pound():
    assert strip_fillers("one pound of beef") == "1 lb of beef"


def test_normalize_gram():
    assert strip_fillers("fifty grams of sugar") == "50 g of sugar"


def test_normalize_milliliter():
    assert strip_fillers("two hundred milliliters of stock") == "200 ml of stock"


# --- spoken URLs ---


def test_normalize_spoken_dot():
    assert strip_fillers("visit fnyc dot com for more") == "visit fnyc.com for more"


def test_normalize_spoken_dot_chained():
    assert strip_fillers("go to www dot fnyc dot com") == "go to www.fnyc.com"


def test_normalize_letter_sequence():
    assert strip_fillers("visit w n y c dot com") == "visit wnyc.com"


# --- repeated punctuation ---


def test_normalize_repeated_periods():
    assert strip_fillers("well...... that is interesting.") == ". that is interesting."


def test_normalize_repeated_exclamation():
    assert strip_fillers("that is great!!! really.") == "that is great! really."


def test_normalize_repeated_question():
    assert strip_fillers("what?? really.") == "what? really."


# --- fused phone numbers ---


def test_phone_re_matches_standard():
    assert _PHONE_RE.search("718-497-2128")


def test_phone_re_matches_fused():
    assert _PHONE_RE.search("718497-2128")


def test_phone_re_matches_no_separators():
    assert _PHONE_RE.search("7184972128")


# --- phrase normalization ---


def test_pain_in_the_butt():
    assert strip_fillers("it's a real pain in the butt.") == "it's a real pain."


def test_pain_in_the_ass():
    assert strip_fillers("that was a pain in the ass to fix.") == "that was a pain to fix."


def test_some_of_them():
    assert strip_fillers("some of them will work.") == "some will work."


def test_some_of_those():
    assert strip_fillers("some of those are fine.") == "some are fine."


# --- its stutter ---


def test_its_stutter_false_start():
    assert strip_fillers("it's a it's problem.") == "it's problem."


def test_its_it_end():
    assert strip_fillers("the answer is it's it.") == "the answer is it's."


# --- like / so opener ---


def test_strips_like_opener():
    assert strip_fillers("Like, you just add salt.") == "You just add salt."


def test_strips_so_opener():
    assert strip_fillers("So, the technique is this.") == "The technique is this."


def test_strips_stacked_so_like():
    assert strip_fillers("So like basically, it works.") == "It works."


# --- I think ---


def test_strips_i_think_prefix():
    assert strip_fillers("I think it tastes better cold.") == "It tastes better cold."


def test_preserves_i_think_about():
    assert strip_fillers("I think about that often.") == "I think about that often."


# --- The problem is ---


def test_strips_the_problem_is():
    assert strip_fillers("The problem is the salt level.") == "Problem is the salt level."


# --- yeah / anyway / nice openers ---


def test_strips_yeah_opener():
    assert strip_fillers("Yeah, that is correct.") == "That is correct."


def test_strips_yeah_opener_no_comma():
    assert strip_fillers("Yeah it works.") == "It works."


def test_strips_anyway_opener():
    assert strip_fillers("Anyway, let's continue.") == "Let's continue."


def test_removes_standalone_nice():
    assert strip_fillers("Great technique. Nice. Now cook.") == "Great technique. Cook."


def test_strips_nice_with_comma():
    assert strip_fillers("Nice, that worked out well.") == "That worked out well."


def test_removes_standalone_anyway():
    assert strip_fillers("It cooked well. Anyway. We added salt.") == "It cooked well. We added salt."


# --- by the way (sentence start) ---


def test_strips_by_the_way_sentence_start():
    assert strip_fillers("By the way, it shrinks when cooked.") == "It shrinks when cooked."


# --- let me just say / tell you ---


def test_strips_let_me_just_say_start():
    assert strip_fillers("Let me just say, this is important.") == "This is important."


def test_strips_let_me_just_tell_you_start():
    assert strip_fillers("Let me just tell you, it matters.") == "It matters."


def test_strips_let_me_just_say_parenthetical():
    assert strip_fillers("The result, let me just say, was perfect.") == "The result was perfect."


# --- very very ---


def test_very_very_no_comma():
    assert strip_fillers("it cooks very very quickly.") == "it cooks very quickly."


def test_very_very_with_comma():
    assert strip_fillers("it is very, very good.") == "it is very good."


# --- I mean (no trailing comma) ---


def test_strips_i_mean_no_comma():
    assert strip_fillers("I mean you could eat it.") == "You could eat it."


def test_strips_i_mean_no_comma_i_subject():
    assert strip_fillers("I mean I could eat it.") == "I could eat it."


def test_preserves_i_mean_it():
    assert strip_fillers("I mean it when I say this.") == "I mean it when I say this."


# --- well / now / listen openers ---


def test_strips_well_opener():
    assert strip_fillers("Well, the technique is this.") == "The technique is this."


def test_strips_now_opener():
    assert strip_fillers("Now, the important thing is salt.") == "The important thing is salt."


def test_strips_listen_opener():
    assert strip_fillers("Listen, your pretzels must be twisted.") == "Your pretzels must be twisted."


# --- you see parenthetical ---


def test_strips_you_see_parenthetical():
    assert strip_fillers("Chocolate, you see, needs roasting.") == "Chocolate needs roasting."


def test_strips_you_see_after_comma():
    assert strip_fillers("It ferments, you see. The temperature rises.") == "It ferments. The temperature rises."


# --- cooking issues / great / said standalone ---


def test_removes_cooking_issues_standalone():
    assert strip_fillers("Your pretzels must be twisted. Cooking issues.") == "Your pretzels must be twisted."


def test_removes_great_standalone():
    assert strip_fillers("That worked. Great. Now add salt.") == "That worked. Add salt."


def test_removes_said_standalone():
    assert strip_fillers("I don't believe it. Said. Moving on.") == "I don't believe it. Moving on."


def test_removes_what_do_you_think_standalone():
    assert (
        strip_fillers("That's the technique. What do you think? Add the salt.") == "That's the technique. Add the salt."
    )


# --- yeah great / yeah nice combos ---


def test_strips_yeah_great():
    assert strip_fillers("The crust formed. Yeah, great. Add the salt.") == "The crust formed. Add the salt."


def test_strips_yeah_nice():
    assert strip_fillers("It worked. Yeah, nice. Continue cooking.") == "It worked. Continue cooking."
