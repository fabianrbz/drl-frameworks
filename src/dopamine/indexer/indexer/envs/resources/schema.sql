--
-- PostgreSQL database dump
--

-- Dumped from database version 11.1
-- Dumped by pg_dump version 11.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: customer; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.customer (
    c_custkey integer NOT NULL,
    c_name character varying(25),
    c_address character varying(40),
    c_nationkey bigint NOT NULL,
    c_phone character(15),
    c_acctbal numeric,
    c_mktsegment character(10),
    c_comment character varying(117)
);


ALTER TABLE public.customer OWNER TO fabian;

--
-- Name: customer_c_custkey_seq; Type: SEQUENCE; Schema: public; Owner: fabian
--

CREATE SEQUENCE public.customer_c_custkey_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.customer_c_custkey_seq OWNER TO fabian;

--
-- Name: customer_c_custkey_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: fabian
--

ALTER SEQUENCE public.customer_c_custkey_seq OWNED BY public.customer.c_custkey;


--
-- Name: lineitem; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.lineitem (
    l_orderkey bigint NOT NULL,
    l_partkey bigint NOT NULL,
    l_suppkey bigint NOT NULL,
    l_linenumber integer NOT NULL,
    l_quantity numeric,
    l_extendedprice numeric,
    l_discount numeric,
    l_tax numeric,
    l_returnflag character(1),
    l_linestatus character(1),
    l_shipdate date,
    l_commitdate date,
    l_receiptdate date,
    l_shipinstruct character(25),
    l_shipmode character(10),
    l_comment character varying(44)
);


ALTER TABLE public.lineitem OWNER TO fabian;

--
-- Name: nation; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.nation (
    n_nationkey integer NOT NULL,
    n_name character(25),
    n_regionkey bigint NOT NULL,
    n_comment character varying(152)
);


ALTER TABLE public.nation OWNER TO fabian;

--
-- Name: nation_n_nationkey_seq; Type: SEQUENCE; Schema: public; Owner: fabian
--

CREATE SEQUENCE public.nation_n_nationkey_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nation_n_nationkey_seq OWNER TO fabian;

--
-- Name: nation_n_nationkey_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: fabian
--

ALTER SEQUENCE public.nation_n_nationkey_seq OWNED BY public.nation.n_nationkey;


--
-- Name: orders; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.orders (
    o_orderkey integer NOT NULL,
    o_custkey bigint NOT NULL,
    o_orderstatus character(1),
    o_totalprice numeric,
    o_orderdate date,
    o_orderpriority character(15),
    o_clerk character(15),
    o_shippriority integer,
    o_comment character varying(79)
);


ALTER TABLE public.orders OWNER TO fabian;

--
-- Name: orders_o_orderkey_seq; Type: SEQUENCE; Schema: public; Owner: fabian
--

CREATE SEQUENCE public.orders_o_orderkey_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.orders_o_orderkey_seq OWNER TO fabian;

--
-- Name: orders_o_orderkey_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: fabian
--

ALTER SEQUENCE public.orders_o_orderkey_seq OWNED BY public.orders.o_orderkey;


--
-- Name: part; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.part (
    p_partkey integer NOT NULL,
    p_name character varying(55),
    p_mfgr character(25),
    p_brand character(10),
    p_type character varying(25),
    p_size integer,
    p_container character(10),
    p_retailprice numeric,
    p_comment character varying(23)
);


ALTER TABLE public.part OWNER TO fabian;

--
-- Name: part_p_partkey_seq; Type: SEQUENCE; Schema: public; Owner: fabian
--

CREATE SEQUENCE public.part_p_partkey_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.part_p_partkey_seq OWNER TO fabian;

--
-- Name: part_p_partkey_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: fabian
--

ALTER SEQUENCE public.part_p_partkey_seq OWNED BY public.part.p_partkey;


--
-- Name: partsupp; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.partsupp (
    ps_partkey bigint NOT NULL,
    ps_suppkey bigint NOT NULL,
    ps_availqty integer,
    ps_supplycost numeric,
    ps_comment character varying(199)
);


ALTER TABLE public.partsupp OWNER TO fabian;

--
-- Name: region; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.region (
    r_regionkey integer NOT NULL,
    r_name character(25),
    r_comment character varying(152)
);


ALTER TABLE public.region OWNER TO fabian;

--
-- Name: region_r_regionkey_seq; Type: SEQUENCE; Schema: public; Owner: fabian
--

CREATE SEQUENCE public.region_r_regionkey_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.region_r_regionkey_seq OWNER TO fabian;

--
-- Name: region_r_regionkey_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: fabian
--

ALTER SEQUENCE public.region_r_regionkey_seq OWNED BY public.region.r_regionkey;


--
-- Name: supplier; Type: TABLE; Schema: public; Owner: fabian
--

CREATE TABLE public.supplier (
    s_suppkey integer NOT NULL,
    s_name character(25),
    s_address character varying(40),
    s_nationkey bigint NOT NULL,
    s_phone character(15),
    s_acctbal numeric,
    s_comment character varying(101)
);


ALTER TABLE public.supplier OWNER TO fabian;

--
-- Name: supplier_s_suppkey_seq; Type: SEQUENCE; Schema: public; Owner: fabian
--

CREATE SEQUENCE public.supplier_s_suppkey_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.supplier_s_suppkey_seq OWNER TO fabian;

--
-- Name: supplier_s_suppkey_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: fabian
--

ALTER SEQUENCE public.supplier_s_suppkey_seq OWNED BY public.supplier.s_suppkey;


--
-- Name: customer c_custkey; Type: DEFAULT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.customer ALTER COLUMN c_custkey SET DEFAULT nextval('public.customer_c_custkey_seq'::regclass);


--
-- Name: nation n_nationkey; Type: DEFAULT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.nation ALTER COLUMN n_nationkey SET DEFAULT nextval('public.nation_n_nationkey_seq'::regclass);


--
-- Name: orders o_orderkey; Type: DEFAULT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.orders ALTER COLUMN o_orderkey SET DEFAULT nextval('public.orders_o_orderkey_seq'::regclass);


--
-- Name: part p_partkey; Type: DEFAULT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.part ALTER COLUMN p_partkey SET DEFAULT nextval('public.part_p_partkey_seq'::regclass);


--
-- Name: region r_regionkey; Type: DEFAULT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.region ALTER COLUMN r_regionkey SET DEFAULT nextval('public.region_r_regionkey_seq'::regclass);


--
-- Name: supplier s_suppkey; Type: DEFAULT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.supplier ALTER COLUMN s_suppkey SET DEFAULT nextval('public.supplier_s_suppkey_seq'::regclass);


--
-- Name: customer customer_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.customer
    ADD CONSTRAINT customer_pkey PRIMARY KEY (c_custkey);


--
-- Name: lineitem lineitem_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.lineitem
    ADD CONSTRAINT lineitem_pkey PRIMARY KEY (l_orderkey, l_linenumber);


--
-- Name: nation nation_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.nation
    ADD CONSTRAINT nation_pkey PRIMARY KEY (n_nationkey);


--
-- Name: orders orders_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_pkey PRIMARY KEY (o_orderkey);


--
-- Name: part part_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.part
    ADD CONSTRAINT part_pkey PRIMARY KEY (p_partkey);


--
-- Name: partsupp partsupp_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.partsupp
    ADD CONSTRAINT partsupp_pkey PRIMARY KEY (ps_partkey, ps_suppkey);


--
-- Name: region region_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.region
    ADD CONSTRAINT region_pkey PRIMARY KEY (r_regionkey);


--
-- Name: supplier supplier_pkey; Type: CONSTRAINT; Schema: public; Owner: fabian
--

ALTER TABLE ONLY public.supplier
    ADD CONSTRAINT supplier_pkey PRIMARY KEY (s_suppkey);


--
-- PostgreSQL database dump complete
--

