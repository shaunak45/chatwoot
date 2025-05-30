require 'classifier-reborn'

require_relative 'intent_examples/refund_request_examples'
require_relative 'intent_examples/order_status_examples'
require_relative 'intent_examples/complaint_examples'
require_relative 'intent_examples/greeting_examples'
require_relative 'intent_examples/unknown_examples'

class Messages::IntentDetectionService
  def initialize(message)
    @message = message
  end

  def perform
    classifier = self.class.classifier
    normalized_text = normalize_text(@message.content)

    intent = classifier.classify(normalized_text)
    classifications = classifier.classifications(normalized_text)

    # Convert log scores to probabilities via softmax
    probabilities = softmax(classifications.values)

    # Find predicted intent and its probability (confidence)
    intent_index = classifications.values.index(classifications.values.max)
    intent = classifications.keys[intent_index]
    confidence = probabilities[intent_index]
    confidence = 0.0 if confidence.nil? || confidence.nan?

    update_content_attributes(intent, confidence)
  end

  private

  def normalize_text(text)
    text.to_s.downcase.strip
  end

  def update_content_attributes(intent, confidence)
    attrs = @message.content_attributes || {}
    attrs['intent'] = intent
    attrs['confidence'] = confidence.round(3)

    @message.update!(content_attributes: attrs)
  end

  def self.classifier
    @classifier ||= build_classifier
  end

  def self.build_classifier
    classifier = ClassifierReborn::Bayes.new('refund_request', 'order_status', 'complaint', 'greeting', 'unknown')

    IntentExamples::RefundRequestExamples::EXAMPLES.each { |text| classifier.train('refund_request', text.downcase) }
    IntentExamples::OrderStatusExamples::EXAMPLES.each { |text| classifier.train('order_status', text.downcase) }
    IntentExamples::ComplaintExamples::EXAMPLES.each { |text| classifier.train('complaint', text.downcase) }
    IntentExamples::GreetingExamples::EXAMPLES.each { |text| classifier.train('greeting', text.downcase) }
    IntentExamples::UnknownExamples::EXAMPLES.each { |text| classifier.train('unknown', text.downcase) }

    classifier
  end

  def softmax(logits)
    max_logit = logits.max
    exps = logits.map { |l| Math.exp(l - max_logit) }

    sum_exps = exps.sum

    if sum_exps.zero? || sum_exps.nan?
      size = logits.size
      return Array.new(size, 1.0 / size)
    end

    exps.map { |e| e / sum_exps }
  end
end
