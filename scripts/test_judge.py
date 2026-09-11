from eval.judge import LLMJudge

judge = LLMJudge()

triv_reply = "Thanks for reaching out to Apple Support! We would be glad to help. Please send us a Direct Message with your device model and details so we can assist: https://twitter.com/messages/compose?recipient_id=AppleSupport"
score_triv = judge.score_reply("My iPhone 8 battery drains fast", triv_reply, "battery_performance", ["Suggest checking Battery Health", "Inquire about iOS version"])

ai_reply = "We would be glad to look into this with you. Please check Battery Health in Settings to view maximum capacity, and DM us your iOS version: https://t.co/GDrqU22YpT"
score_ai = judge.score_reply("My iPhone 8 battery drains fast", ai_reply, "battery_performance", ["Suggest checking Battery Health", "Inquire about iOS version"])

print("Trivial Baseline Judge Score:", score_triv["overall_score"], score_triv)
print("AI Agent Judge Score:", score_ai["overall_score"], score_ai)
