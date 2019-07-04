-- phpMyAdmin SQL Dump
-- version 4.8.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Jul 03, 2019 at 11:21 AM
-- Server version: 5.7.23
-- PHP Version: 7.2.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `allan`
--

-- --------------------------------------------------------

--
-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE IF NOT EXISTS `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE IF NOT EXISTS `auth_group_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissions_group_id_b120cbf9` (`group_id`),
  KEY `auth_group_permissions_permission_id_84c5c92e` (`permission_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE IF NOT EXISTS `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  KEY `auth_permission_content_type_id_2f476e4b` (`content_type_id`)
) ENGINE=MyISAM AUTO_INCREMENT=41 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `auth_permission`
--

INSERT INTO `auth_permission` (`id`, `name`, `content_type_id`, `codename`) VALUES
(1, 'Can add log entry', 1, 'add_logentry'),
(2, 'Can change log entry', 1, 'change_logentry'),
(3, 'Can delete log entry', 1, 'delete_logentry'),
(4, 'Can view log entry', 1, 'view_logentry'),
(5, 'Can add permission', 2, 'add_permission'),
(6, 'Can change permission', 2, 'change_permission'),
(7, 'Can delete permission', 2, 'delete_permission'),
(8, 'Can view permission', 2, 'view_permission'),
(9, 'Can add group', 3, 'add_group'),
(10, 'Can change group', 3, 'change_group'),
(11, 'Can delete group', 3, 'delete_group'),
(12, 'Can view group', 3, 'view_group'),
(13, 'Can add user', 4, 'add_user'),
(14, 'Can change user', 4, 'change_user'),
(15, 'Can delete user', 4, 'delete_user'),
(16, 'Can view user', 4, 'view_user'),
(17, 'Can add content type', 5, 'add_contenttype'),
(18, 'Can change content type', 5, 'change_contenttype'),
(19, 'Can delete content type', 5, 'delete_contenttype'),
(20, 'Can view content type', 5, 'view_contenttype'),
(21, 'Can add session', 6, 'add_session'),
(22, 'Can change session', 6, 'change_session'),
(23, 'Can delete session', 6, 'delete_session'),
(24, 'Can view session', 6, 'view_session'),
(25, 'Can add chat', 7, 'add_chat'),
(26, 'Can change chat', 7, 'change_chat'),
(27, 'Can delete chat', 7, 'delete_chat'),
(28, 'Can view chat', 7, 'view_chat'),
(29, 'Can add domain', 8, 'add_domain'),
(30, 'Can change domain', 8, 'change_domain'),
(31, 'Can delete domain', 8, 'delete_domain'),
(32, 'Can view domain', 8, 'view_domain'),
(33, 'Can add label', 9, 'add_label'),
(34, 'Can change label', 9, 'change_label'),
(35, 'Can delete label', 9, 'delete_label'),
(36, 'Can view label', 9, 'view_label'),
(37, 'Can add chat line', 10, 'add_chatline'),
(38, 'Can change chat line', 10, 'change_chatline'),
(39, 'Can delete chat line', 10, 'delete_chatline'),
(40, 'Can view chat line', 10, 'view_chatline');

-- --------------------------------------------------------

--
-- Table structure for table `auth_user`
--

DROP TABLE IF EXISTS `auth_user`;
CREATE TABLE IF NOT EXISTS `auth_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(30) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `auth_user`
--

INSERT INTO `auth_user` (`id`, `password`, `last_login`, `is_superuser`, `username`, `first_name`, `last_name`, `email`, `is_staff`, `is_active`, `date_joined`) VALUES
(1, 'pbkdf2_sha256$150000$vQnQmQKWMkB6$9fWwrF3VMh5SVFbiyIgFpuapeXFqsuT/NaQ6VnIzGPE=', '2019-07-02 13:42:57.851042', 1, 'admin', '', '', 'admin@admin.com', 1, 1, '2019-06-30 18:10:59.357745'),
(2, 'pbkdf2_sha256$150000$r8zngCJEM39m$Di/z0FHUDYnoYFE8ArRM6JWZPLED/fpD5V7Rjthz0Fw=', NULL, 1, 'allan', '', '', 'allan@allan.com', 1, 1, '2019-07-03 08:57:43.467416');

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_groups`
--

DROP TABLE IF EXISTS `auth_user_groups`;
CREATE TABLE IF NOT EXISTS `auth_user_groups` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_user_id_6a12ed8b` (`user_id`),
  KEY `auth_user_groups_group_id_97559544` (`group_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_user_permissions`
--

DROP TABLE IF EXISTS `auth_user_user_permissions`;
CREATE TABLE IF NOT EXISTS `auth_user_user_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permissions_user_id_a95ead1b` (`user_id`),
  KEY `auth_user_user_permissions_permission_id_1fbb5f2c` (`permission_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `conversation_editor_chat`
--

DROP TABLE IF EXISTS `conversation_editor_chat`;
CREATE TABLE IF NOT EXISTS `conversation_editor_chat` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(100) NOT NULL,
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  `created_user_id` int(11) DEFAULT NULL,
  `domain_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `conversation_editor_chat_created_user_id_a4928d71` (`created_user_id`),
  KEY `conversation_editor_chat_domain_id_00b7c3e8` (`domain_id`)
) ENGINE=MyISAM AUTO_INCREMENT=23 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `conversation_editor_chat`
--

INSERT INTO `conversation_editor_chat` (`id`, `title`, `created`, `updated`, `created_user_id`, `domain_id`) VALUES
(22, 'chat_1562152303', '2019-07-03 11:11:43.454931', '2019-07-03 11:11:43.454931', 2, 1);

-- --------------------------------------------------------

--
-- Table structure for table `conversation_editor_chatline`
--

DROP TABLE IF EXISTS `conversation_editor_chatline`;
CREATE TABLE IF NOT EXISTS `conversation_editor_chatline` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `message` varchar(1024) NOT NULL,
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  `chat_id` int(11) NOT NULL,
  `created_user_id` int(11) DEFAULT NULL,
  `label_id` int(11) NOT NULL,
  `parent_id` int(11) DEFAULT NULL,
  `human` tinyint(1) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `conversation_editor_chatline_chat_id_ec6b0f9e` (`chat_id`),
  KEY `conversation_editor_chatline_created_user_id_c36451cd` (`created_user_id`),
  KEY `conversation_editor_chatline_label_id_b70a6a94` (`label_id`),
  KEY `conversation_editor_chatline_parent_id_id_2d38cb33` (`parent_id`)
) ENGINE=MyISAM AUTO_INCREMENT=26 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `conversation_editor_chatline`
--

INSERT INTO `conversation_editor_chatline` (`id`, `message`, `created`, `updated`, `chat_id`, `created_user_id`, `label_id`, `parent_id`, `human`) VALUES
(22, 'Buna User', '2019-07-03 11:11:43.463715', '2019-07-03 11:11:43.463715', 22, 2, 1, NULL, 0),
(23, 'Buna Oana', '2019-07-03 11:11:43.477379', '2019-07-03 11:11:43.477379', 22, 2, 1, NULL, 1),
(24, 'Cu ce te pot ajuta?', '2019-07-03 11:11:43.487139', '2019-07-03 11:11:43.487139', 22, 2, 8, NULL, 0),
(25, 'Ma doare rau capul', '2019-07-03 11:11:43.510562', '2019-07-03 11:11:43.510562', 22, 2, 9, NULL, 1);

-- --------------------------------------------------------

--
-- Table structure for table `conversation_editor_domain`
--

DROP TABLE IF EXISTS `conversation_editor_domain`;
CREATE TABLE IF NOT EXISTS `conversation_editor_domain` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `domainName` varchar(50) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `conversation_editor_domain`
--

INSERT INTO `conversation_editor_domain` (`id`, `domainName`) VALUES
(1, 'Health'),
(2, 'Realestate');

-- --------------------------------------------------------

--
-- Table structure for table `conversation_editor_label`
--

DROP TABLE IF EXISTS `conversation_editor_label`;
CREATE TABLE IF NOT EXISTS `conversation_editor_label` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `domain_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `conversation_editor_label_domain_id_7d2a6eec` (`domain_id`)
) ENGINE=MyISAM AUTO_INCREMENT=12 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `conversation_editor_label`
--

INSERT INTO `conversation_editor_label` (`id`, `name`, `domain_id`) VALUES
(1, 'salut', 1),
(2, 'neutru', 1),
(3, 'salut', 2),
(4, 'neutru', 2),
(5, 'nmi', 2),
(7, 'salutare', 2),
(8, 'question', 1),
(9, 'sanatate', 1),
(10, 'ajutor', 1);

-- --------------------------------------------------------

--
-- Table structure for table `conversation_editor_label_chat`
--

DROP TABLE IF EXISTS `conversation_editor_label_chat`;
CREATE TABLE IF NOT EXISTS `conversation_editor_label_chat` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `label_id` int(11) NOT NULL,
  `chat_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `conversation_editor_label_chat_label_id_chat_id_19805b9e_uniq` (`label_id`,`chat_id`),
  KEY `conversation_editor_label_chat_label_id_91f7b2ac` (`label_id`),
  KEY `conversation_editor_label_chat_chat_id_9ac4ecef` (`chat_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE IF NOT EXISTS `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) UNSIGNED NOT NULL,
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE IF NOT EXISTS `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=MyISAM AUTO_INCREMENT=11 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_content_type`
--

INSERT INTO `django_content_type` (`id`, `app_label`, `model`) VALUES
(1, 'admin', 'logentry'),
(2, 'auth', 'permission'),
(3, 'auth', 'group'),
(4, 'auth', 'user'),
(5, 'contenttypes', 'contenttype'),
(6, 'sessions', 'session'),
(7, 'conversation_editor', 'chat'),
(8, 'conversation_editor', 'domain'),
(9, 'conversation_editor', 'label'),
(10, 'conversation_editor', 'chatline');

-- --------------------------------------------------------

--
-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE IF NOT EXISTS `django_migrations` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=24 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_migrations`
--

INSERT INTO `django_migrations` (`id`, `app`, `name`, `applied`) VALUES
(1, 'contenttypes', '0001_initial', '2019-06-30 09:25:52.623668'),
(2, 'auth', '0001_initial', '2019-06-30 09:25:52.693469'),
(3, 'admin', '0001_initial', '2019-06-30 09:25:52.918955'),
(4, 'admin', '0002_logentry_remove_auto_add', '2019-06-30 09:25:52.973139'),
(5, 'admin', '0003_logentry_add_action_flag_choices', '2019-06-30 09:25:52.978995'),
(6, 'contenttypes', '0002_remove_content_type_name', '2019-06-30 09:25:53.011204'),
(7, 'auth', '0002_alter_permission_name_max_length', '2019-06-30 09:25:53.028771'),
(8, 'auth', '0003_alter_user_email_max_length', '2019-06-30 09:25:53.044892'),
(9, 'auth', '0004_alter_user_username_opts', '2019-06-30 09:25:53.050748'),
(10, 'auth', '0005_alter_user_last_login_null', '2019-06-30 09:25:53.066363'),
(11, 'auth', '0006_require_contenttypes_0002', '2019-06-30 09:25:53.067339'),
(12, 'auth', '0007_alter_validators_add_error_messages', '2019-06-30 09:25:53.073195'),
(13, 'auth', '0008_alter_user_username_max_length', '2019-06-30 09:25:53.092800'),
(14, 'auth', '0009_alter_user_last_name_max_length', '2019-06-30 09:25:53.109308'),
(15, 'auth', '0010_alter_group_name_max_length', '2019-06-30 09:25:53.124923'),
(16, 'auth', '0011_update_proxy_permissions', '2019-06-30 09:25:53.131755'),
(17, 'conversation_editor', '0001_initial', '2019-06-30 09:25:53.226932'),
(18, 'conversation_editor', '0002_chatline', '2019-06-30 09:25:53.419890'),
(19, 'conversation_editor', '0003_auto_20190628_1532', '2019-06-30 09:25:53.547745'),
(20, 'conversation_editor', '0004_remove_chat_location', '2019-06-30 09:25:53.567265'),
(21, 'sessions', '0001_initial', '2019-06-30 09:25:53.579953'),
(22, 'conversation_editor', '0005_auto_20190701_1258', '2019-07-01 09:58:27.784632'),
(23, 'conversation_editor', '0006_chatline_human', '2019-07-03 09:31:10.105858');

-- --------------------------------------------------------

--
-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
CREATE TABLE IF NOT EXISTS `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_session`
--

INSERT INTO `django_session` (`session_key`, `session_data`, `expire_date`) VALUES
('b7zc1t7atxb46l5f3nt42egq4pl5b7yl', 'YWRkZGFmNWU0YzEwMTc1Y2JlNjViZGEwZjFlY2RhNDU2NjdjNTMwNDp7Il9hdXRoX3VzZXJfaWQiOiIxIiwiX2F1dGhfdXNlcl9iYWNrZW5kIjoiZGphbmdvLmNvbnRyaWIuYXV0aC5iYWNrZW5kcy5Nb2RlbEJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI2M2VhZTk5ODkxYzAzZDBjZmI3M2Y2YmNiMDFlNzViMTliM2ZkOTY3In0=', '2019-07-16 13:42:57.852994');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
