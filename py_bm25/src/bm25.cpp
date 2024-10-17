#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>
#include "bm25.h"


namespace BM25 {
    BM25_::BM25_(
        const std::vector< std::vector<std::string> > & corpus,
        double k1,
        double b,
        bool invert_index
    ) :
    k1(k1),
    b(b), 
    invert_index(invert_index) {
        corpus_size = corpus.size();
        doc_len.reserve(corpus_size);
        if (invert_index) {
            build_inverted_index(corpus);
        }
        else {
            index.reserve(corpus_size);
            build_index(corpus);
        }
        idfs.reserve(nd.size());
        Eigen::ArrayXd doc_len_arr = Eigen::ArrayXi::Map(&doc_len[0], corpus_size).cast<double>();
        cst = 1 - b + b * (doc_len_arr / avgdl);
    }

    void BM25_::build_index(const std::vector< std::vector<std::string> > & corpus) {
        int total_tokens(0);
        for (const std::vector<std::string> & tokens : corpus) {
            const int tokens_size = tokens.size();
            doc_len.push_back(tokens_size);
            total_tokens += tokens_size;

            std::unordered_map<std::string, int> freqs;
            for (const std::string & token : tokens) {
                ++freqs[token];
            }
            index.push_back(std::move(freqs));

            for (const std::pair<const std::string, const int> & token : index.back()) {
                ++nd[token.first];
            }
        }
        avgdl = static_cast<double>(total_tokens) / corpus_size;
    }

    void BM25_::build_inverted_index(const std::vector < std::vector<std::string> > & corpus) {
        int total_tokens(0);
        
        for (int i = 0; i < corpus.size(); ++i) {
            const std::vector<std::string> & doc = corpus[i];
            std::unordered_set<std::string> seen_tokens;
            const int tokens_size = doc.size();
            doc_len.push_back(tokens_size);
            total_tokens += tokens_size;

            for (const std::string & token: doc) {
                if (seen_tokens.insert(token).second) {
                    ++nd[token];
                }

                if (inverted_index.find(token) == inverted_index.end()){
                    inverted_index[token] = std::vector<int>(corpus.size(), 0);
                }

                ++inverted_index[token][i];
            }
        }
        avgdl = static_cast<double>(total_tokens) / corpus_size;
    }

    Eigen::ArrayXd BM25_::get_scores(std::vector<std::string> & query) {
        if (invert_index) {
            return get_scores_from_inverted_index(query);
        }
        else {
            return get_scores_from_index(query);
        }
    }

    Eigen::ArrayXXd BM25_::get_scores_batch(std::vector< std::vector<std::string> > & queries) {
        const int num_queries = queries.size();
        Eigen::ArrayXXd scores(num_queries, corpus_size);

        if (invert_index)
        {
            #pragma omp parallel for
            for (int i = 0; i < num_queries; ++i) {
                std::vector<std::string>& query = queries[i];
                scores.row(i) = get_scores_from_inverted_index(query);
            }
        }
        else {
            #pragma omp parallel for
            for (int i = 0; i < num_queries; ++i) {
                std::vector<std::string>& query = queries[i];
                scores.row(i) = get_scores_from_index(query);
            }
        }
            
        return scores;
    }

    int BM25_::get_corpus_size() const {return corpus_size;}
    double BM25_::get_avgdl() const {return avgdl;}
    bool BM25_::get_invert_index() const {return invert_index;}
    double BM25_::get_k1() const { return k1;}
    double BM25_::get_b() const {return b;}
    void BM25_::set_k1(double k1) {this->k1 = k1;}
    void BM25_::set_b(double b) {this->b = b;}
    std::unordered_map<std::string, double>& BM25_::get_idfs() {return idfs;}
    std::vector<int>& BM25_::get_doc_len() {return doc_len;}
    std::vector< std::unordered_map<std::string, int> >& BM25_::get_index() {
        return index;
    }
    std::unordered_map<std::string, std::vector<int>> & BM25_::get_inverted_index() {
        return inverted_index;
    }

    BM25Okapi_::BM25Okapi_(
        const std::vector< std::vector<std::string> > & corpus, 
        double k1, 
        double b, 
        double epsilon,
        bool invert_index
    ) : 
    BM25::BM25_(corpus, k1, b, invert_index), epsilon(epsilon) {
        calc_idf(nd);
        cst = k1 * cst;
    }

    void BM25Okapi_::calc_idf(const std::unordered_map<std::string, int> & nd) {
        double idf_sum = 0;
        std::vector<std::string> negative_idfs;
        for (const std::pair<const std::string, const int> & token : nd) {
            const double idf = std::log(corpus_size - token.second + 0.5) - std::log(token.second + 0.5);
            idfs[token.first] = idf;
            idf_sum += idf;
            if (idf < 0) {
                negative_idfs.push_back(token.first);
            }
        }

        const double avg_idf = idf_sum / idfs.size();
        const double eps = epsilon * avg_idf;

        for (const std::string & token : negative_idfs) {
            idfs[token] = eps;
        }
    }

    double BM25Okapi_::get_epsilon() const {return epsilon;}
    void BM25Okapi_::set_epsilon(double epsilon) {this->epsilon = epsilon;}

    Eigen::ArrayXd BM25Okapi_::get_scores_from_index(std::vector<std::string> & query) {
        Eigen::ArrayXd scores_arr = Eigen::ArrayXd::Zero(corpus_size);
        Eigen::ArrayXd freqs_arr(corpus_size);
        std::vector<int> freqs(corpus_size);
        
        for (int i = 0; i < query.size(); ++i) {
            std::string q = query[i];
            const auto idf_iter = idfs.find(q);
            if (idf_iter == idfs.end()) continue;
            const double idf = idf_iter->second;

            #pragma omp parallel for
            for (int j = 0; j < corpus_size; ++j) {
                std::unordered_map<std::string, int>& doc_freq = index[j];
                const auto doc_freq_iter = doc_freq.find(q);
                freqs[j] = (doc_freq_iter != doc_freq.end() ? doc_freq_iter->second : 0);
            }

            freqs_arr = Eigen::ArrayXi::Map(&freqs[0], corpus_size).cast<double>();
            scores_arr += idf * (freqs_arr * (k1 + 1) / (freqs_arr + cst));
        }

        return scores_arr;
    }   

    Eigen::ArrayXd BM25Okapi_::get_scores_from_inverted_index(std::vector<std::string> & query) {
        const int num_tokens = query.size();
        std::unordered_map<std::string, double>::const_iterator idf_it;
        Eigen::ArrayXXd q_freqs(num_tokens, corpus_size);
        Eigen::ArrayXd q_idfs(num_tokens);

        for (int i = 0; i < num_tokens; ++i) {
            const std::string & token = query[i];
            idf_it = idfs.find(token);
            if (idf_it != idfs.end()){
                q_idfs(i) = idf_it->second;
                q_freqs.row(i) = Eigen::ArrayXi::Map(&inverted_index[token][0], corpus_size).cast<double>();
            }
            else {
                q_idfs(i) = 0.0;
                q_freqs.row(i).setZero();
            }
        }

        Eigen::ArrayXXd nomin = q_freqs * (k1 + 1);
        Eigen::ArrayXXd denomin = q_freqs + cst.transpose().replicate(num_tokens, 1);
        Eigen::ArrayXd scores_arr = ((nomin / denomin).colwise() * q_idfs).colwise().sum();

        return scores_arr;
    }

    BM25L_::BM25L_(
        const std::vector< std::vector<std::string> > & corpus, 
        double k1, 
        double b, 
        double delta,
        bool invert_index
    ) : 
    BM25::BM25_(corpus, k1, b, invert_index), delta(delta) {
        calc_idf(nd);
    }

    void BM25L_::calc_idf(const std::unordered_map<std::string, int> & nd) {
        for (const std::pair<const std::string, const int> & token : nd) {
            const double idf = std::log(corpus_size + 1) - std::log(token.second + 0.5);
            idfs[token.first] = idf;
        }
    }

    double BM25L_::get_delta() const {return delta;}
    void BM25L_::set_delta(double delta) {this->delta = delta;}

    Eigen::ArrayXd BM25L_::get_scores_from_index(std::vector<std::string> & query) {
        Eigen::ArrayXd scores_arr = Eigen::ArrayXd::Zero(corpus_size);
        Eigen::ArrayXd tmp_cst(corpus_size);
        Eigen::ArrayXd freqs_arr(corpus_size);
        std::vector<int> freqs(corpus_size);
        
        for (int i = 0; i < query.size(); ++i) {
            std::string q = query[i];
            const auto idf_iter = idfs.find(q);
            if (idf_iter == idfs.end()) continue;
            const double idf = idf_iter->second;

            #pragma omp parallel for
            for (int j = 0; j < corpus_size; ++j) {
                std::unordered_map<std::string, int>& doc_freq = index[j];
                const auto doc_freq_iter = doc_freq.find(q);
                freqs[j] = (doc_freq_iter != doc_freq.end() ? doc_freq_iter->second : 0);
            }

            freqs_arr = Eigen::ArrayXi::Map(&freqs[0], corpus_size).cast<double>();
            tmp_cst = freqs_arr / cst;
            scores_arr += idf * (k1 + 1) * (tmp_cst + delta) / (k1 + tmp_cst + delta);
        }

        return scores_arr;
    }   

    Eigen::ArrayXd BM25L_::get_scores_from_inverted_index(std::vector<std::string> & query) {
        const int num_tokens = query.size();
        std::unordered_map<std::string, double>::const_iterator idf_it;
        Eigen::ArrayXd q_freqs(corpus_size);
        Eigen::ArrayXXd q_csts(num_tokens, corpus_size);
        Eigen::ArrayXd q_idfs(num_tokens);

        for (int i = 0; i < num_tokens; ++i) {
            const std::string & token = query[i];
            idf_it = idfs.find(token);
            if (idf_it != idfs.end()){
                q_idfs(i) = idf_it->second;
                q_freqs = Eigen::ArrayXi::Map(&inverted_index[token][0], corpus_size).cast<double>();
                q_csts.row(i) = q_freqs / cst;
            }
            else {
                q_idfs(i) = 0.0;
                q_csts.row(i).setZero();
            }
        }

        Eigen::ArrayXXd nomin = (q_idfs.replicate(1, corpus_size) * (k1 + 1)) * (q_csts + delta);
        Eigen::ArrayXXd denomin = (k1 + q_csts + delta);
        Eigen::ArrayXd scores_arr = (nomin / denomin).colwise().sum();

        return scores_arr;
    }

    BM25Plus_::BM25Plus_(
        const std::vector< std::vector<std::string> > & corpus, 
        double k1, 
        double b, 
        double delta,
        bool invert_index
    ) : 
    BM25::BM25_(corpus, k1, b, invert_index), delta(delta) {
        calc_idf(nd);
        cst = k1 * cst;
    }

    void BM25Plus_::calc_idf(const std::unordered_map<std::string, int> & nd) {
        for (const std::pair<const std::string, const int> & token : nd) {
            const double idf = std::log(corpus_size + 1) - std::log(token.second);
            idfs[token.first] = idf;
        }
    }

    double BM25Plus_::get_delta() const {return delta;}
    void BM25Plus_::set_delta(double delta) {this->delta = delta;}

    Eigen::ArrayXd BM25Plus_::get_scores_from_index(std::vector<std::string> & query) {
        Eigen::ArrayXd scores_arr = Eigen::ArrayXd::Zero(corpus_size);
        Eigen::ArrayXd freqs_arr(corpus_size);
        std::vector<int> freqs(corpus_size);
        
        for (int i = 0; i < query.size(); ++i) {
            std::string q = query[i];
            const auto idf_iter = idfs.find(q);
            if (idf_iter == idfs.end()) continue;
            const double idf = idf_iter->second;

            #pragma omp parallel for
            for (int j = 0; j < corpus_size; ++j) {
                std::unordered_map<std::string, int>& doc_freq = index[j];
                const auto doc_freq_iter = doc_freq.find(q);
                freqs[j] = (doc_freq_iter != doc_freq.end() ? doc_freq_iter->second : 0);
            }

            freqs_arr = Eigen::ArrayXi::Map(&freqs[0], corpus_size).cast<double>();
            scores_arr += idf * ( ( ((k1 + 1) * freqs_arr) / (cst + freqs_arr) ) + delta );
        }

        return scores_arr;
    }   

    Eigen::ArrayXd BM25Plus_::get_scores_from_inverted_index(std::vector<std::string> & query) {
        const int num_tokens = query.size();
        std::unordered_map<std::string, double>::const_iterator idf_it;
        Eigen::ArrayXXd q_freqs(num_tokens, corpus_size);
        Eigen::ArrayXd q_idfs(num_tokens);

        for (int i = 0; i < num_tokens; ++i) {
            const std::string & token = query[i];
            idf_it = idfs.find(token);
            if (idf_it != idfs.end()){
                q_idfs(i) = idf_it->second;
                q_freqs.row(i) = Eigen::ArrayXi::Map(&inverted_index[token][0], corpus_size).cast<double>();
            }
            else {
                q_idfs(i) = 0.0;
                q_freqs.row(i).setZero();
            }
        }

        Eigen::ArrayXXd tmp = ( 
            ((k1 + 1) * q_freqs) / (cst.transpose().replicate(num_tokens, 1) + q_freqs) 
        ) + delta;
        Eigen::ArrayXd scores_arr = (q_idfs.replicate(1, corpus_size) * tmp).colwise().sum();
        
        return scores_arr;
    }

}
