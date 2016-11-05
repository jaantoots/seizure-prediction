#!/usr/bin/env ruby

require 'CSV'

rows = CSV.read('train_and_test_data_labels_safe.csv')
rows.shift

rows.each do |row|
  name = row[0]
  # Extract label from name
  /^\d+_\d+_(?<name_label>\d+)\.mat$/ =~ name
  # True label
  label = row[1]
  # Check if any mismatches
  if name_label && name_label != label
    puts('==> Label mismatch: ' + name)
  end
end
