#!/usr/bin/ruby
require 'trollop'
require 'json'
require 'byebug'
require 'set'
require 'pathname'

$num_failed = 0
$num_read = 0

def read_sample(id, fname)
  sample = {:id => id}
  begin
    $num_read += 1
    if $num_read % 1000 == 0
      puts "\033[1A [Failed//Read]: [#{$num_failed} // #{$num_read}]"
    end
    sample.merge(
      JSON.parse(File.read(fname))
      .map{ |k, v| [k.to_sym, v] }
      .to_h
    )
  rescue
    $num_failed += 1
    sample
  end
end

def group_samples(samples, aspect)
  samples.group_by{ |sample| Set.new(sample[aspect]) }
end

def join_splits(splits)
  big = []
  small = []
  splits.each{ |new_big, new_small|
    big.concat(new_big)
    small.concat(new_small)
  }
  return big, small
end

def split_by_rating(samples, split_fraction)
  leftovers = []
  by_rating = samples.group_by{ |sample| sample[:rating] }
  splits_for_each_rating = by_rating.map{ |k, v|
    split_by_genre(v, split_fraction, leftovers)
  }
  final_split = join_splits(splits_for_each_rating)
  #deal with leftovers:
  puts "ratings leftovers: #{leftovers.size}"
  puts "genre leftovers: #{$genre_leftovers.size}"
  puts "cat leftovers: #{$cat_leftovers.size}"
  join_splits([final_split, simple_split(leftovers, split_fraction)])
end

$genre_leftovers = Set.new()
def split_by_genre(samples, split_fraction, leftovers)
  sub_leftovers = []
  by_genre = group_samples(samples, :genre)
  splits_for_each_genre = by_genre.map{ |k, v|
    split_by_category(v, split_fraction, sub_leftovers)
  }
  final_split = join_splits(splits_for_each_genre)
  # deal with leftovers:
  if belongs_in_leftovers(sub_leftovers) then
    leftovers.concat(sub_leftovers)
    final_split
  else
    $genre_leftovers = $genre_leftovers.merge(sub_leftovers)
    join_splits([final_split, simple_split(sub_leftovers, split_fraction)])
  end
end

$cat_leftovers = Set.new()
def split_by_category(samples, split_fraction, leftovers)
  sub_leftovers = []
  by_category = group_samples(samples, :category)
  splits_for_each_category = by_category.map{ |k, v|
    split_by_words(v, split_fraction, sub_leftovers)
  }
  final_split = join_splits(splits_for_each_category)
  # deal with leftovers:
  if belongs_in_leftovers(sub_leftovers) then
    leftovers.concat(sub_leftovers)
    final_split
  else
    $cat_leftovers = $cat_leftovers.merge(sub_leftovers)
    join_splits([final_split, simple_split(sub_leftovers, split_fraction)])  
  end
end

def belongs_in_leftovers(samples)
  samples.size < 20
end

def split_by_words(samples, split_fraction, leftovers)
  big = []
  small = []
  if belongs_in_leftovers(samples)
    leftovers.concat(samples)
  else
    # bin into each order of magnitude
    by_oom = samples.group_by{ |sample| 
      begin
        Math.log(sample[:words], 10).floor
      rescue
        byebug
        puts "32"
      end
    }
    subsplits = by_oom.map{ |k, v|
      if belongs_in_leftovers(v)
        leftovers.concat(v)
        [[], []]
      else
        simple_split(v, split_fraction)
      end
    }
    return join_splits(subsplits)
  end
  [big, small]
end

def simple_split(samples, split_fraction)
  first_n = (split_fraction * samples.size).ceil
  shuf = samples.shuffle
  [shuf[0...first_n], shuf[first_n..-1]]
end

def get_opts
  Trollop::options do
    opt :tok_dir, "directory containing tokenized files", :short => 't', :type => String
    opt :id_list, "file containing ids to run on", :short => 'l', :type => String
    opt :info_dir, "directory containing info files", :short => 'i', :type => String
    opt :out_prefix, "prefix for out files", :short => 'o', :type => String
    opt :split_fraction, "fraction to put in first split", :short => 'f', :type => Float, :default => 0.8
    opt :out_fname_big, "fname for large out file", :short => 'b', :type => String, :default => "train"
    opt :out_fname_small, "fname for small out file", :short => 's', :type => String, :default => "dev"
  end
end

def stats_from_samples(samples)
  {"words" => samples.map{ |sample| sample[:words] }.sum}
end

def write_samples_and_stats_to_file(samples, fname, sample_dir)
  norm_dir = sample_dir.chomp("/")
  ids = samples.map{ |sample| sample[:id] }
  open(fname, 'w') { |f|
    f.puts(ids.map{ |id| "#{norm_dir}/#{id}.npy" } )
  }
  open(fname + "_stats.json", 'w') { |s_f|
    s_f.write(stats_from_samples(samples).to_json)
  }
end

if __FILE__ == $PROGRAM_NAME
  opts = get_opts()

  info_dir = opts[:info_dir].chomp("/")
  out_prefix = opts[:out_prefix]
  split_fraction = opts[:split_fraction]
  tok_dir = opts[:tok_dir].chomp("/") + "/"

  sample_ids = 
    if opts[:id_list_given] then
      File.readlines(opts[:id_list]).map{ |line| 
        Pathname.new(line.strip).basename.to_s.gsub(/\..*/, "")
      }
    else
      `find #{tok_dir} -type f -name "*.tok"`
        .split("\n")
        .map{ |fname| fname.strip.chomp(".tok").gsub(tok_dir, "") }
    end

  sample_ids = sample_ids.sort#[0..1000]

  puts "Reading Samples!\n\n"
  samples = sample_ids
    .map{ |id| read_sample(id, "#{info_dir}/#{id}.info") }
    .select{ |sample| sample.size > 1 }

  samples = samples.select{ |sample| sample[:words] > 10 }
  byebug

  by_rating = samples.group_by{ |sample| sample[:rating] }
  oom = samples.map{ |sample| 
    begin
      Math.log(sample[:words], 10).floor
    rescue
      0 
    end
  }.group_by{ |a| a }
    .map{ |k, v| [k, v.size]}
    .sort{ |a, b| a[0] <=> b[0] }
  by_genre = group_samples(samples, :genre)
  by_cat = group_samples(samples, :category)

  genre_cnts = by_genre.each_with_object(Hash.new(0)){ |tup, cnts|
    genres, samps = tup
    cnts[genres.size] += samps.size
  }
  cat_cnts = by_cat.each_with_object(Hash.new(0)){ |tup, cnts|
    genres, samps = tup
    cnts[genres.size] += samps.size
  }

  final_split = split_by_rating(samples, split_fraction)
  big, small = final_split

#  write_samples_and_stats_to_file(big, out_prefix + opts[:out_fname_big], tok_dir)
#  write_samples_and_stats_to_file(small, out_prefix + opts[:out_fname_small], tok_dir)
end

